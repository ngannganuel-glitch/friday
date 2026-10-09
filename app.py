"""StockLab VN - He thong phan tich co hoi dau tu co phieu.
Chay: streamlit run app.py
"""
import io, datetime as dt, xml.etree.ElementTree as ET
import numpy as np, pandas as pd, requests, streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="StockLab VN", page_icon="📈", layout="wide")

PEERS = {
    "Ngân hàng": ["VCB", "BID", "CTG", "TCB", "MBB", "ACB", "VPB"],
    "Công nghệ": ["FPT", "CMG", "ELC"],
    "Bán lẻ": ["MWG", "PNJ", "DGW", "FRT"],
    "Thực phẩm - Đồ uống": ["VNM", "MSN", "SAB", "KDC"],
    "Thép": ["HPG", "HSG", "NKG"],
    "Bất động sản": ["VHM", "VIC", "NVL", "KDH", "DXG"],
    "Chứng khoán": ["SSI", "VND", "HCM", "VCI"],
    "Dầu khí - Điện": ["GAS", "PLX", "POW", "PVD"],
}
POS = ["tăng", "lãi", "kỷ lục", "vượt", "tích cực", "mua ròng", "khởi sắc", "cổ tức", "tăng trưởng", "bứt phá"]
NEG = ["giảm", "lỗ", "bán ròng", "vi phạm", "thanh tra", "khởi tố", "nợ xấu", "sụt", "cảnh báo", "đình chỉ", "phạt"]


# ---------------------------- 1. THU THẬP DỮ LIỆU ----------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_prices(sym, years=3, is_index=False):
    end = dt.date.today()
    start = end - dt.timedelta(days=int(365 * years))
    df = None
    try:  # Nguồn 1: vnstock (dữ liệu sàn Việt Nam)
        from vnstock import Vnstock
        raw = Vnstock().stock(symbol=sym, source="VCI").quote.history(
            start=str(start), end=str(end), interval="1D")
        raw.columns = [c.lower() for c in raw.columns]
        raw["time"] = pd.to_datetime(raw["time"])
        df = raw.set_index("time")[["open", "high", "low", "close", "volume"]]
    except Exception:
        df = None
    if (df is None or df.empty) and not is_index:
        try:  # Nguồn 2 dự phòng: Yahoo Finance
            import yfinance as yf
            y = yf.download(sym + ".VN", start=start, end=end, auto_adjust=True, progress=False)
            if isinstance(y.columns, pd.MultiIndex):
                y.columns = y.columns.get_level_values(0)
            y.columns = [c.lower() for c in y.columns]
            df = y[["open", "high", "low", "close", "volume"]]
            df.index = pd.to_datetime(df.index)
        except Exception:
            df = None
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.dropna().sort_index()
    if not is_index and df["close"].median() > 1000:  # thống nhất đơn vị: nghìn đồng
        df[["open", "high", "low", "close"]] /= 1000
    return df


@st.cache_data(ttl=86400, show_spinner=False)
def get_fund(sym):
    keys = ["longName", "sector", "industry", "marketCap", "trailingPE", "priceToBook",
            "returnOnEquity", "returnOnAssets", "profitMargins", "debtToEquity",
            "dividendYield", "revenueGrowth", "earningsGrowth", "trailingEps"]
    try:
        import yfinance as yf
        info = yf.Ticker(sym + ".VN").info
        return {k: info.get(k) for k in keys}
    except Exception:
        return {k: None for k in keys}


@st.cache_data(ttl=1800, show_spinner=False)
def get_news(sym, n=12):
    url = f"https://news.google.com/rss/search?q=cổ+phiếu+{sym}&hl=vi&gl=VN&ceid=VN:vi"
    out = []
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        for it in ET.fromstring(r.content).findall(".//item")[:n]:
            title = it.findtext("title") or ""
            low = title.lower()
            s = sum(w in low for w in POS) - sum(w in low for w in NEG)
            out.append({"Tiêu đề": title, "Nguồn/Ngày": (it.findtext("pubDate") or "")[:25],
                        "Link": it.findtext("link"), "Điểm": s})
    except Exception:
        pass
    return pd.DataFrame(out)


@st.cache_data(ttl=3600, show_spinner=False)
def get_macro():
    res = {}
    try:
        import yfinance as yf
        for name, tk in {"USD/VND": "VND=X", "Lợi suất TPCP Mỹ 10Y (%)": "^TNX", "Vàng (USD/oz)": "GC=F"}.items():
            d = yf.download(tk, period="1y", progress=False, auto_adjust=True)
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            s = d["Close"].dropna()
            res[name] = (float(s.iloc[-1]), float(s.iloc[-1] / s.iloc[0] - 1))
    except Exception:
        pass
    return res


# ---------------------------- 2. PHÂN TÍCH ----------------------------
def add_ind(df):
    df = df.copy(); c = df["close"]
    for w in (20, 50, 200):
        df[f"sma{w}"] = c.rolling(w).mean()
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14).mean()
    df["rsi"] = 100 - 100 / (1 + up / dn)
    df["macd"] = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    df["sig"] = df["macd"].ewm(span=9).mean()
    m, s = c.rolling(20).mean(), c.rolling(20).std()
    df["bbu"], df["bbl"] = m + 2 * s, m - 2 * s
    return df


def risk_stats(df, idx):
    r = np.log(df["close"]).diff().dropna()
    ret, vol = r.mean() * 252, r.std() * np.sqrt(252)
    out = {"Lợi suất BQ năm": ret, "Biến động năm": vol,
           "Sharpe (rf=4%)": (ret - 0.04) / vol if vol else np.nan,
           "Sụt giảm tối đa": (df["close"] / df["close"].cummax() - 1).min(),
           "Lợi suất 1 năm": df["close"].iloc[-1] / df["close"].iloc[max(-252, -len(df))] - 1}
    if not idx.empty:
        j = pd.concat([r, np.log(idx["close"]).diff()], axis=1, join="inner").dropna()
        if len(j) > 30:
            out["Beta so với VN-Index"] = j.cov().iloc[0, 1] / j.iloc[:, 1].var()
    return out


def fnum(x, pct=False, d=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x * 100:.{d}f}%" if pct else f"{x:,.{d}f}"


def scoring(df, f, news_df, rk):
    L = df.iloc[-1]; t = 0
    t += 15 if L.close > L.sma50 else 0
    t += 10 if L.sma50 > L.sma200 else 0
    t += 10 if 40 < L.rsi < 70 else (6 if L.rsi <= 30 else 0)
    t += 5 if L.macd > L.sig else 0
    pe, roe = f.get("trailingPE"), f.get("returnOnEquity")
    fa = 0
    fa += 20 if pe and 0 < pe < 15 else (12 if pe and pe < 25 else (4 if pe else 10))
    fa += 15 if roe and roe > .15 else (10 if roe and roe > .10 else (3 if roe else 7))
    v, dd = rk["Biến động năm"], rk["Sụt giảm tối đa"]
    ri = (8 if v < .3 else 5 if v < .45 else 2) + (7 if dd > -.2 else 4 if dd > -.35 else 1)
    ns = 5 + int(np.clip(news_df["Điểm"].sum(), -5, 5)) if not news_df.empty else 5
    total = t + fa + ri + ns
    rec = "TÍCH CỰC – có thể tích lũy" if total >= 70 else "TRUNG LẬP – theo dõi" if total >= 50 else "THẬN TRỌNG"
    return {"Kỹ thuật (/40)": t, "Cơ bản (/35)": fa, "Rủi ro (/15)": ri, "Tin tức (/10)": ns}, total, rec


def peer_table(sym, peers, years=1):
    rows = []
    for p in dict.fromkeys([sym] + peers):
        d = get_prices(p, years)
        if d.empty:
            continue
        r = np.log(d["close"]).diff().dropna()
        fu = get_fund(p)
        rows.append({"Mã": p, "Giá": d["close"].iloc[-1],
                     "LS 1 năm": d["close"].iloc[-1] / d["close"].iloc[0] - 1,
                     "Biến động": r.std() * np.sqrt(252), "P/E": fu.get("trailingPE"),
                     "ROE": fu.get("returnOnEquity")})
    return pd.DataFrame(rows)


def narrative(sym, df, f, rk, sc, total, rec, macro):
    L = df.iloc[-1]
    trend = "tăng" if L.close > L.sma50 > L.sma200 else "giảm" if L.close < L.sma50 < L.sma200 else "đi ngang/phân hóa"
    rsi = "quá mua" if L.rsi > 70 else "quá bán" if L.rsi < 30 else "trung tính"
    out = [f"Cổ phiếu {sym} đang trong xu hướng {trend} (giá {L.close:,.2f} nghìn đồng; SMA50 {L.sma50:,.2f}; SMA200 {L.sma200:,.2f}). "
           f"RSI(14) = {L.rsi:.1f} ({rsi}); MACD {'trên' if L.macd > L.sig else 'dưới'} đường tín hiệu.",
           f"Lợi suất 1 năm {fnum(rk['Lợi suất 1 năm'], True)}, biến động {fnum(rk['Biến động năm'], True)}/năm, "
           f"sụt giảm tối đa {fnum(rk['Sụt giảm tối đa'], True)}, Sharpe {fnum(rk['Sharpe (rf=4%)'])}.",
           f"Định giá: P/E {fnum(f.get('trailingPE'))}, P/B {fnum(f.get('priceToBook'))}, ROE {fnum(f.get('returnOnEquity'), True)}."]
    if macro:
        out.append("Bối cảnh vĩ mô: " + "; ".join(f"{k} {v[0]:,.2f} ({v[1]:+.1%} trong 1 năm)" for k, v in macro.items()) + ".")
    out.append(f"Tổng điểm hệ thống: {total}/100 → {rec}. Đây là kết quả định lượng tham khảo, không phải khuyến nghị đầu tư.")
    return out


# ---------------------------- 3. BIỂU ĐỒ ----------------------------
def main_chart(df, sym):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[.6, .2, .2], vertical_spacing=.03)
    fig.add_trace(go.Candlestick(x=df.index, open=df.open, high=df.high, low=df.low, close=df.close,
                                 name=sym, increasing_line_color="#00a651", decreasing_line_color="#e53935"), 1, 1)
    for w, col in ((20, "#ff9800"), (50, "#2196f3"), (200, "#9c27b0")):
        fig.add_trace(go.Scatter(x=df.index, y=df[f"sma{w}"], name=f"SMA{w}", line=dict(width=1, color=col)), 1, 1)
    fig.add_trace(go.Scatter(x=df.index, y=df.bbu, name="BB trên", line=dict(width=.5, dash="dot", color="gray")), 1, 1)
    fig.add_trace(go.Scatter(x=df.index, y=df.bbl, name="BB dưới", line=dict(width=.5, dash="dot", color="gray")), 1, 1)
    fig.add_trace(go.Bar(x=df.index, y=df.volume, name="KLGD", marker_color="#90a4ae"), 2, 1)
    fig.add_trace(go.Scatter(x=df.index, y=df.rsi, name="RSI", line=dict(color="#673ab7")), 3, 1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    fig.update_layout(height=700, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h"))
    return fig


# ---------------------------- 4. XUẤT PDF ----------------------------
def make_pdf(sym, df, f, rk, sc, total, rec, news_df, peers_df, macro):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    pdfmetrics.registerFont(TTFont("VN", matplotlib.get_data_path() + "/fonts/ttf/DejaVuSans.ttf"))
    H = ParagraphStyle("h", fontName="VN", fontSize=14, textColor=colors.HexColor("#0d47a1"), spaceAfter=6, spaceBefore=10)
    N = ParagraphStyle("n", fontName="VN", fontSize=9.5, leading=14)
    T = ParagraphStyle("t", fontName="VN", fontSize=20, textColor=colors.HexColor("#0d47a1"))

    def table(rows, widths=None):
        t = Table(rows, colWidths=widths)
        t.setStyle(TableStyle([("FONTNAME", (0, 0), (-1, -1), "VN"), ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                               ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e3f2fd")),
                               ("GRID", (0, 0), (-1, -1), .3, colors.grey)]))
        return t

    fig, ax = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    d = df.tail(250)
    ax[0].plot(d.index, d.close, color="#0d47a1", label="Giá")
    ax[0].plot(d.index, d.sma50, label="SMA50"); ax[0].plot(d.index, d.sma200, label="SMA200")
    ax[0].fill_between(d.index, d.bbl, d.bbu, alpha=.1); ax[0].legend(); ax[0].set_title(f"{sym} – 1 năm gần nhất")
    ax[1].plot(d.index, d.rsi, color="purple"); ax[1].axhline(70, c="r", ls="--"); ax[1].axhline(30, c="g", ls="--")
    ax[1].set_ylabel("RSI"); fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=130); plt.close(fig); buf.seek(0)

    S = [Paragraph(f"BÁO CÁO PHÂN TÍCH CỔ PHIẾU {sym}", T),
         Paragraph(f"Ngày lập: {dt.date.today():%d/%m/%Y} | {f.get('longName') or ''} | {f.get('sector') or ''}", N),
         Paragraph("1. Kết luận tổng quan", H)]
    S += [Paragraph(x, N) for x in narrative(sym, df, f, rk, sc, total, rec, macro)]
    S += [Paragraph("2. Điểm đánh giá", H), table([["Hạng mục", "Điểm"]] + [[k, v] for k, v in sc.items()] + [["TỔNG (/100)", total]]),
          Paragraph("3. Phân tích kỹ thuật", H), Image(buf, width=17 * cm, height=10.4 * cm),
          Paragraph("4. Rủi ro", H),
          table([["Chỉ tiêu", "Giá trị"]] + [[k, fnum(v, pct="Sharpe" not in k and "Beta" not in k)] for k, v in rk.items()]),
          Paragraph("5. Chỉ số cơ bản", H),
          table([["Chỉ tiêu", "Giá trị"], ["P/E", fnum(f.get("trailingPE"))], ["P/B", fnum(f.get("priceToBook"))],
                 ["ROE", fnum(f.get("returnOnEquity"), True)], ["Biên LN", fnum(f.get("profitMargins"), True)],
                 ["Nợ/Vốn CSH", fnum(f.get("debtToEquity"))], ["Tăng trưởng DT", fnum(f.get("revenueGrowth"), True)]])]
    if not peers_df.empty:
        rows = [["Mã", "Giá", "LS 1 năm", "Biến động", "P/E", "ROE"]]
        for _, r in peers_df.iterrows():
            rows.append([r["Mã"], fnum(r["Giá"]), fnum(r["LS 1 năm"], True), fnum(r["Biến động"], True), fnum(r["P/E"]), fnum(r["ROE"], True)])
        S += [Paragraph("6. So sánh cùng ngành", H), table(rows)]
    if not news_df.empty:
        S += [Paragraph("7. Tin tức gần đây", H)] + [Paragraph("• " + t, N) for t in news_df["Tiêu đề"].head(8)]
    S += [Spacer(1, 10), Paragraph("Nguồn: vnstock/VCI, Yahoo Finance, Google News. Báo cáo tự động, chỉ mang tính tham khảo.", N)]
    out = io.BytesIO(); SimpleDocTemplate(out, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm).build(S)
    return out.getvalue()


# ---------------------------- 5. GIAO DIỆN ----------------------------
st.markdown("""<style>.hdr{background:#0d47a1;color:#fff;padding:12px 20px;border-radius:6px;font-size:22px;font-weight:700}
.sub{font-size:13px;font-weight:400;opacity:.85}</style>
<div class="hdr">📈 StockLab VN <span class="sub">– Phân tích vĩ mô · ngành · cơ hội đầu tư cổ phiếu</span></div>""",
            unsafe_allow_html=True)

with st.sidebar:
    sym = st.text_input("Mã cổ phiếu", "FPT").upper().strip()
    years = st.slider("Số năm dữ liệu", 1, 5, 3)
    grp = next((g for g, l in PEERS.items() if sym in l), None)
    peers_in = st.text_input("Mã cùng ngành (cách nhau dấu phẩy)", ",".join(p for p in PEERS.get(grp, []) if p != sym))
    go_btn = st.button("Phân tích", type="primary", use_container_width=True)

if go_btn or "ready" not in st.session_state:
    st.session_state["ready"] = True
    with st.spinner("Đang thu thập & phân tích dữ liệu..."):
        px = get_prices(sym, years)
        if px.empty:
            st.error("Không lấy được dữ liệu. Kiểm tra mã/kết nối mạng.")
            st.stop()
        idx = get_prices("VNINDEX", years, is_index=True)
        df = add_ind(px); f = get_fund(sym); news = get_news(sym); macro = get_macro()
        rk = risk_stats(df, idx); sc, total, rec = scoring(df, f, news, rk)
        peers = [p.strip().upper() for p in peers_in.split(",") if p.strip()]
        pt = peer_table(sym, peers)
    st.session_state["res"] = (sym, df, idx, f, news, macro, rk, sc, total, rec, pt, grp)

sym, df, idx, f, news, macro, rk, sc, total, rec, pt, grp = st.session_state["res"]
chg = df.close.iloc[-1] / df.close.iloc[-2] - 1
c = st.columns(5)
c[0].metric(f"{sym} (nghìn đ)", f"{df.close.iloc[-1]:,.2f}", f"{chg:+.2%}")
c[1].metric("P/E", fnum(f.get("trailingPE")))
c[2].metric("ROE", fnum(f.get("returnOnEquity"), True))
c[3].metric("Điểm hệ thống", f"{total}/100")
c[4].metric("Đánh giá", rec.split(" – ")[0])

t1, t2, t3, t4, t5, t6 = st.tabs(["Tổng quan", "Biểu đồ kỹ thuật", "Cơ bản & Rủi ro", "Ngành", "Vĩ mô & Tin tức", "Xuất PDF"])
with t1:
    for x in narrative(sym, df, f, rk, sc, total, rec, macro):
        st.write("• " + x)
    st.bar_chart(pd.Series(sc))
with t2:
    st.plotly_chart(main_chart(df.tail(365), sym), use_container_width=True)
with t3:
    a, b = st.columns(2)
    a.subheader("Chỉ số cơ bản")
    a.table(pd.DataFrame({"Giá trị": {"P/E": fnum(f.get("trailingPE")), "P/B": fnum(f.get("priceToBook")),
                                      "ROE": fnum(f.get("returnOnEquity"), True), "ROA": fnum(f.get("returnOnAssets"), True),
                                      "Biên LN": fnum(f.get("profitMargins"), True), "Nợ/Vốn CSH": fnum(f.get("debtToEquity")),
                                      "Tỷ suất cổ tức": fnum(f.get("dividendYield"), True)}}))
    b.subheader("Rủi ro")
    b.table(pd.DataFrame({"Giá trị": {k: fnum(v, pct=("Sharpe" not in k and "Beta" not in k)) for k, v in rk.items()}}))
with t4:
    st.subheader(f"So sánh cùng ngành {('– ' + grp) if grp else ''}")
    if not pt.empty:
        st.dataframe(pt.style.format({"Giá": "{:,.2f}", "LS 1 năm": "{:.1%}", "Biến động": "{:.1%}", "P/E": "{:.1f}", "ROE": "{:.1%}"}), use_container_width=True)
        st.bar_chart(pt.set_index("Mã")["LS 1 năm"])
with t5:
    a, b = st.columns(2)
    a.subheader("Vĩ mô thị trường")
    for k, v in macro.items():
        a.metric(k, f"{v[0]:,.2f}", f"{v[1]:+.1%} (1 năm)")
    if not idx.empty:
        a.line_chart(idx["close"].tail(365))
    b.subheader("Tin tức & cảm xúc")
    if news.empty:
        b.info("Không lấy được tin tức.")
    for _, r in news.iterrows():
        b.markdown(f"{'🟢' if r['Điểm'] > 0 else '🔴' if r['Điểm'] < 0 else '⚪'} [{r['Tiêu đề']}]({r['Link']})")
with t6:
    st.write("Báo cáo PDF tự động (tiếng Việt, có biểu đồ, bảng điểm, so sánh ngành, tin tức).")
    if st.button("Tạo báo cáo PDF"):
        pdf = make_pdf(sym, df, f, rk, sc, total, rec, news, pt, macro)
        st.download_button("⬇️ Tải PDF", pdf, f"BaoCao_{sym}_{dt.date.today():%Y%m%d}.pdf", "application/pdf")
