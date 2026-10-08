
import os, base64, io, json
import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output, State, dash_table, no_update, ctx
import plotly.express as px
import plotly.graph_objects as go
import anthropic

from src.data_processing import prepare_data
from src.ml_models import load_models, predict_borrower
from src.forecasting import monthly_payment_forecast

BASE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(BASE,"data","borrowers.csv")
MODEL_DIR = os.path.join(BASE,"models")

# Reads the ANTHROPIC_API_KEY environment variable automatically.
# Set it before running: export ANTHROPIC_API_KEY="sk-ant-..."
# Built lazily so the dashboard still starts without a key - only the
# Assistant tab needs one.
CLAUDE_MODEL = "claude-sonnet-5"
_claude_client = None

def claude():
    global _claude_client
    if _claude_client is None:
        _claude_client = anthropic.Anthropic()
    return _claude_client

app = Dash(__name__, suppress_callback_exceptions=True, title="AI Loan Affordability")
server = app.server

def fmt_money(x):
    if x is None or pd.isna(x): return "—"
    return f"TZS {x:,.0f}"

def metric_card(title, value, subtitle=""):
    return html.Div([html.Div(title, className="metric-title"),
                     html.Div(value, className="metric-value"),
                     html.Div(subtitle, className="metric-sub")], className="metric-card")

def load_default():
    return prepare_data(pd.read_csv(DEFAULT_DATA))

DF = load_default()
MODELS = load_models(MODEL_DIR)

app.layout = html.Div([
    dcc.Store(id="data-store"),
    html.Div(className="header", children=[
        html.Div([
            html.H1("AI Borrower Affordability & Repayment Analytics"),
            html.P("Decision-support dashboard for affordability, repayment performance and risk forecasting.")
        ]),
        html.Div([
            dcc.Upload(id="upload-data", children=html.Button("Upload CSV Dataset", className="upload-btn"),
                       multiple=False),
            html.Button("Use Demo Dataset", id="demo-btn", className="secondary-btn")
        ], className="header-actions")
    ]),

    dcc.Tabs(id="tabs", value="overview", children=[
        dcc.Tab(label="Overview", value="overview"),
        dcc.Tab(label="Affordability", value="affordability"),
        dcc.Tab(label="AI Assessment", value="assessment"),
        dcc.Tab(label="Repayment & Forecast", value="forecast"),
        dcc.Tab(label="Loan Simulator", value="simulator"),
        dcc.Tab(label="Data", value="data"),
        dcc.Tab(label="Assistant", value="assistant")
    ], className="tabs"),

    html.Div(id="page-content")
], className="app")

def overview_page(df):
    k = {
        "borrowers": len(df),
        "avg_income": df.monthly_income.mean(),
        "avg_loan": df.loan_amount.mean(),
        "repayment": df.repayment_efficiency.mean(),
        "score": df.affordability_score.mean(),
        "default": df.defaulted.mean()*100
    }
    d = df.copy()
    d["month"] = pd.to_datetime(d.application_date, errors="coerce").dt.to_period("M").astype(str)
    trend = d.groupby("month", as_index=False).agg(amount_repaid=("amount_repaid","sum"), amount_due=("amount_due","sum"))
    p1 = px.line(trend, x="month", y=["amount_repaid","amount_due"], markers=True,
                 labels={"value":"Amount (TZS)","month":"Month","variable":"Metric"},
                 title="Monthly Payment Trend")
    p2 = px.histogram(df, x="monthly_income", nbins=35, title="Income Distribution",
                      labels={"monthly_income":"Monthly Income (TZS)"})
    p3 = px.scatter(df.sample(min(800,len(df)), random_state=2), x="monthly_income", y="loan_amount",
                    color="loan_status", size="repayment_efficiency", hover_data=["borrower_id"],
                    title="Income vs Loan Amount")
    p4 = px.bar(df.groupby("payment_channel", as_index=False)["repayment_efficiency"].mean(),
                x="payment_channel", y="repayment_efficiency", title="Repayment Efficiency by Channel")
    return html.Div([
        html.Div([metric_card("Borrowers",f"{k['borrowers']:,}"),
                  metric_card("Average Income",fmt_money(k["avg_income"])),
                  metric_card("Average Loan",fmt_money(k["avg_loan"])),
                  metric_card("Repayment Efficiency",f"{k['repayment']:.1f}%"),
                  metric_card("Affordability Score",f"{k['score']:.1f}/100"),
                  metric_card("Default Rate",f"{k['default']:.1f}%")], className="metrics"),
        html.Div([dcc.Graph(figure=p1),dcc.Graph(figure=p2)], className="grid-2"),
        html.Div([dcc.Graph(figure=p3),dcc.Graph(figure=p4)], className="grid-2")
    ])

def affordability_page(df):
    p1 = px.scatter(df.sample(min(900,len(df)), random_state=3), x="monthly_income", y="monthly_payment",
                    color="affordability_score", size="loan_amount",
                    hover_data=["borrower_id","dti","pti"],
                    title="Income vs Monthly Loan Payment")
    p2 = px.scatter(df.sample(min(900,len(df)), random_state=4), x="dti", y="affordability_score",
                    color="loan_status", hover_data=["borrower_id","monthly_income","loan_amount"],
                    title="Debt-to-Income Ratio vs Affordability Score")
    p3 = px.histogram(df, x="affordability_score", nbins=30, title="Affordability Score Distribution")
    p4 = px.box(df, x="loan_status", y="disposable_income", title="Disposable Income by Loan Status")
    return html.Div([
        html.Div(className="section-intro", children=[
            html.H2("Borrower Affordability Analysis"),
            html.P("Affordability is evaluated using income, expenses, debt obligations, loan burden, repayment history and derived DTI/PTI indicators.")
        ]),
        html.Div([dcc.Graph(figure=p1),dcc.Graph(figure=p2)], className="grid-2"),
        html.Div([dcc.Graph(figure=p3),dcc.Graph(figure=p4)], className="grid-2")
    ])

def assessment_page(df):
    ids = df.borrower_id.astype(str).tolist()
    return html.Div([
        html.Div(className="section-intro", children=[
            html.H2("AI Borrower Assessment"),
            html.P("Select a borrower to see affordability, repayment probability, risk and recommended lending capacity.")
        ]),
        html.Div([
            html.Label("Borrower"),
            dcc.Dropdown(ids, ids[0] if ids else None, id="borrower-select"),
        ], className="control-card"),
        html.Div(id="assessment-result")
    ])

def forecast_page(df):
    hist = df.copy()
    hist["application_date"] = pd.to_datetime(hist.application_date, errors="coerce")
    monthly = hist.set_index("application_date").resample("MS")["amount_repaid"].sum().reset_index()
    fc = monthly_payment_forecast(df, 6)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly.application_date,y=monthly.amount_repaid,mode="lines+markers",name="Actual repayments"))
    if not fc.empty:
        fig.add_trace(go.Scatter(x=fc.application_date,y=fc.forecast_amount,mode="lines+markers",name="Forecast"))
    fig.update_layout(title="Historical Repayments and Future Forecast",xaxis_title="Month",yaxis_title="TZS")
    eff = px.line(monthly, x="application_date", y="amount_repaid", title="Repayment Trend")
    return html.Div([
        html.Div(className="section-intro", children=[
            html.H2("Repayment Performance & Future Forecast"),
            html.P("Forecast is a transparent trend baseline. For production lending, validate the forecast against out-of-sample historical periods before using it for decisions.")
        ]),
        dcc.Graph(figure=fig),
        dcc.Graph(figure=eff)
    ])

def simulator_page():
    return html.Div([
        html.Div(className="section-intro", children=[
            html.H2("Loan Affordability Simulator"),
            html.P("Test a proposed loan before making a lending decision.")
        ]),
        html.Div([
            html.Div([html.Label("Monthly Income"),dcc.Input(id="sim-income",type="number",value=1000000, className="input")]),
            html.Div([html.Label("Monthly Expenses"),dcc.Input(id="sim-expenses",type="number",value=400000, className="input")]),
            html.Div([html.Label("Existing Debt Payment"),dcc.Input(id="sim-debt",type="number",value=100000, className="input")]),
            html.Div([html.Label("Requested Loan"),dcc.Input(id="sim-loan",type="number",value=1500000, className="input")]),
            html.Div([html.Label("Interest Rate (%)"),dcc.Input(id="sim-rate",type="number",value=18, className="input")]),
            html.Div([html.Label("Term (months)"),dcc.Input(id="sim-term",type="number",value=6, className="input")]),
            html.Button("Assess Loan", id="simulate-btn", className="primary-btn")
        ], className="sim-grid"),
        html.Div(id="simulation-result", className="simulation-result")
    ])

def data_page(df):
    return html.Div([
        html.H2("Dataset Explorer"),
        html.P(f"{len(df):,} records loaded. Upload another CSV from the top-right to analyze your own dataset."),
        dash_table.DataTable(
            data=df.head(100).to_dict("records"),
            columns=[{"name":c,"id":c} for c in df.columns],
            page_size=12, filter_action="native", sort_action="native",
            style_table={"overflowX":"auto"}, style_cell={"padding":"8px","textAlign":"left"}
        )
    ])

def assistant_page():
    return html.Div([
        html.Div(className="section-intro", children=[
            html.H2("AI Assistant"),
            html.P("Ask questions about the loaded dataset, borrower affordability, or how to read the charts.")
        ]),
        dcc.Store(id="chat-history", data=[]),
        html.Div(id="chat-window", className="chat-window"),
        html.Div([
            dcc.Input(id="chat-input", type="text", placeholder="Ask something…",
                      className="input", n_submit=0, debounce=False),
            html.Button("Send", id="chat-send-btn", className="primary-btn")
        ], className="chat-input-row")
    ])

@app.callback(Output("page-content","children"), Input("tabs","value"), State("data-store","data"))
def render_page(tab, stored):
    df = DF if stored is None else prepare_data(pd.DataFrame(stored))
    if tab=="overview": return overview_page(df)
    if tab=="affordability": return affordability_page(df)
    if tab=="assessment": return assessment_page(df)
    if tab=="forecast": return forecast_page(df)
    if tab=="simulator": return simulator_page()
    if tab=="assistant": return assistant_page()
    return data_page(df)

def render_chat_bubbles(history):
    bubbles = []
    for msg in history:
        cls = "chat-bubble user" if msg["role"] == "user" else "chat-bubble bot"
        bubbles.append(html.Div(msg["content"], className=cls))
    if not bubbles:
        bubbles = [html.Div("Ask a question to get started — try \"what does DTI mean here?\"",
                            className="chat-empty")]
    return bubbles

@app.callback(
    Output("chat-window","children"),
    Output("chat-history","data"),
    Output("chat-input","value"),
    Input("chat-send-btn","n_clicks"),
    Input("chat-input","n_submit"),
    State("chat-input","value"),
    State("chat-history","data"),
    State("data-store","data"),
    prevent_initial_call=True
)
def handle_chat(n_clicks, n_submit, message, history, stored):
    history = history or []
    if not message or not message.strip():
        return render_chat_bubbles(history), history, no_update

    df = DF if stored is None else prepare_data(pd.DataFrame(stored))
    context_summary = (
        f"Dataset currently loaded has {len(df):,} borrowers. "
        f"Average monthly income: {df.monthly_income.mean():,.0f} TZS. "
        f"Average loan amount: {df.loan_amount.mean():,.0f} TZS. "
        f"Average affordability score: {df.affordability_score.mean():.1f}/100. "
        f"Default rate: {df.defaulted.mean()*100:.1f}%."
    )

    history.append({"role": "user", "content": message})

    try:
        response = claude().messages.create(
            model=CLAUDE_MODEL,
            max_tokens=600,
            system=(
                "You are an assistant embedded in a loan affordability analytics dashboard. "
                "Answer concisely and help the user interpret their data. "
                f"Current dataset context: {context_summary}"
            ),
            messages=[{"role": m["role"], "content": m["content"]} for m in history]
        )
        reply_text = "".join(
            block.text for block in response.content if block.type == "text"
        ) or "Sorry, I couldn't generate a response."
    except Exception as e:
        reply_text = f"The assistant is unavailable right now ({e})."

    history.append({"role": "assistant", "content": reply_text})
    return render_chat_bubbles(history), history, ""

@app.callback(Output("data-store","data"), Input("upload-data","contents"), Input("demo-btn","n_clicks"), prevent_initial_call=True)
def update_data(contents, clicks):
    from dash import ctx
    if ctx.triggered_id=="demo-btn":
        return DF.to_dict("records")
    if contents:
        try:
            _, content_string = contents.split(",")
            decoded = base64.b64decode(content_string)
            new_df = prepare_data(pd.read_csv(io.StringIO(decoded.decode("utf-8"))))
            return new_df.to_dict("records")
        except Exception:
            return no_update
    return no_update

@app.callback(Output("assessment-result","children"), Input("borrower-select","value"), State("data-store","data"))
def assess(borrower_id, stored):
    df = DF if stored is None else prepare_data(pd.DataFrame(stored))
    if not borrower_id or borrower_id not in df.borrower_id.astype(str).values:
        return html.Div("Select a borrower.", className="alert")
    row = df[df.borrower_id.astype(str)==str(borrower_id)].iloc[0]
    result = predict_borrower(row.to_dict(), MODELS)
    aff = result.get("affordability_probability", float(row["affordability_score"])/100)
    default = result.get("default_probability", 1-row["affordability_score"]/100)
    risk = "LOW" if default < .25 else ("MEDIUM" if default < .50 else "HIGH")
    disposable = float(row.disposable_income)
    max_payment = max(0, disposable*.40)
    max_loan = max_payment * float(row.term_months)
    recommendation = "CONSIDER" if aff >= .60 and default < .35 and disposable > float(row.monthly_payment) else "REVIEW / REDUCE LOAN"
    return html.Div([
        html.Div([metric_card("Affordability Probability",f"{aff*100:.1f}%"),
                  metric_card("Default Risk",f"{default*100:.1f}%"),
                  metric_card("Risk Level",risk),
                  metric_card("Current Affordability Score",f"{row.affordability_score:.1f}/100")], className="metrics"),
        html.Div([
            html.H3("Recommended Lending View"),
            html.P(f"Decision: {recommendation}", className="decision"),
            html.P(f"Recommended maximum payment capacity: {fmt_money(max_payment)}"),
            html.P(f"Indicative maximum loan capacity: {fmt_money(max_loan)}"),
            html.P(f"Current requested loan: {fmt_money(row.loan_amount)}"),
            html.P(f"Disposable income: {fmt_money(disposable)}"),
            html.P(f"DTI: {row.dti*100:.1f}% | PTI: {row.pti*100:.1f}% | Repayment efficiency: {row.repayment_efficiency:.1f}%")
        ], className="result-card")
    ])

@app.callback(Output("simulation-result","children"), Input("simulate-btn","n_clicks"),
              State("sim-income","value"),State("sim-expenses","value"),State("sim-debt","value"),
              State("sim-loan","value"),State("sim-rate","value"),State("sim-term","value"),
              prevent_initial_call=True)
def simulate(n, income, expenses, debt, loan, rate, term):
    vals = [income,expenses,debt,loan,rate,term]
    if any(v is None or v<=0 for v in vals):
        return html.Div("Please enter positive values for all fields.", className="alert")
    r=rate/100/12
    payment=loan*r*(1+r)**term/((1+r)**term-1)
    disposable=income-expenses-debt
    dti=(debt+payment)/income
    pti=payment/income
    score=float(np.clip(100-55*dti-18*(loan/income)+70*.85,0,100))
    if disposable>=payment and dti<=.45 and score>=55:
        decision="AFFORDABLE"
    elif disposable>=payment*.8 and dti<=.60:
        decision="REVIEW / MODERATE"
    else:
        decision="NOT AFFORDABLE"
    return html.Div([
        html.H3(decision),
        html.Div([metric_card("Monthly Payment",fmt_money(payment)),
                  metric_card("Disposable Income",fmt_money(disposable)),
                  metric_card("DTI",f"{dti*100:.1f}%"),
                  metric_card("PTI",f"{pti*100:.1f}%"),
                  metric_card("Affordability Score",f"{score:.1f}/100")], className="metrics"),
        html.P("This simulator is a decision-support prototype; production credit policy should be validated against historical outcomes and regulatory requirements.")
    ])

if __name__=="__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)
 