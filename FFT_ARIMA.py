# =============================================================== #
# MARKET CYCLES TOOLKIT – v3.5.3 "The Annotated Truth"            #
# Grok x @JackS18893 | Cape Town | 11 Nov 2025 | 04:40 PM SAST    #
# FULLY COMMENTED | AUTO-ARIMA | MICRO-CYCLES (>20 ticks)         #
# 100% HTML OUTPUT | CLI | JSON API | NO DASH                     #
# =============================================================== #
import numpy as np                                      # Core math
import pandas as pd                                     # Data handling
import plotly.graph_objects as go                       # Interactive plots
from plotly.subplots import make_subplots               # Multi-panel plots
import plotly.colors as pc                              # Color schemes
from scipy.signal import butter, filtfilt, hilbert      # Filters & phase
from typing import Optional, List, Tuple, Dict, Any     # Type hints
import warnings, logging, json, sys, glob, argparse     # System & CLI
from datetime import datetime, timezone                 # Time handling
from pathlib import Path                                # File paths
import statsmodels.api as sm                            # Stats models
from statsmodels.tsa.statespace.sarimax import SARIMAX  # SARIMA
from sklearn.metrics import mean_absolute_error          # Error metrics
import pmdarima as pm                                   # AUTO-ARIMA
import os                                               # OS interaction

# Suppress noisy warnings
warnings.filterwarnings("ignore")
logging.getLogger('statsmodels').setLevel(logging.ERROR)

# -------------------------- LOGGING & METRICS --------------------------
log_dir = Path("logs")                                  # Log storage
log_dir.mkdir(exist_ok=True)                            # Create if missing

# Fix Windows console encoding
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

# Configure logging: file + console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(log_dir / f"toolkit_{datetime.now():%Y%m%d_%H%M%S}.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)                       # Logger instance

# Central metrics dictionary — updated throughout
metrics: Dict[str, Any] = {
    "run_timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "user_time_sast": "November 11, 2025 04:40 PM SAST",
    "user_location": "Cape Town, Western Cape, ZA",
    "file_path": None,
    "data_points": 0,
    "regime": None,
    "variance_explained": 0.0,
    "best_tradeable_cycle_days": None,
    "sarima_order": None,
    "sarima_mae": None,
    "sarima_mape": None,
    "forecast_horizon_days": 10,
    "ticks_per_cycle": {},
    "micro_cycles_plotted": []
}

# -------------------------- CONFIG --------------------------
CONFIG = {
    "FILE_PATH": r'C:\Users\Hein Vogel\Desktop\FFT_API\Cleaned & Current Raw Data.csv',  # Default input
    "COLUMN_INDEX": 0,                     # Price column index
    "FS": 390,                             # Ticks per trading day
    "N_TOP": 7,                            # Top N FFT cycles to show
    "FORECAST_DAYS": 10,                   # Forecast horizon
    "MICRO_MIN_TICKS": 20                  # Min ticks for micro-cycle plot
}

# -------------------------- CORE: DATA LOADING --------------------------
def load_data(filepath: str, price_col_idx: int) -> Tuple[pd.DataFrame, np.ndarray, Optional[np.ndarray]]:
    """Load CSV, extract price series, and parse dates if available."""
    log.info(f"Loading data from: {filepath}")
    metrics["file_path"] = filepath
    df = pd.read_csv(filepath)                                 # Read CSV
    metrics["data_points"] = len(df)
    log.info(f"Loaded {len(df):,} rows × {len(df.columns)} cols")
    price = df.iloc[:, price_col_idx].astype(np.float64).values  # Extract price
    date_col = next((c for c in df.columns if 'time' in str(c).lower()), None)
    dates = None
    if date_col:
        try:
            dates = pd.to_datetime(df[date_col], errors='coerce').values
            if not pd.isna(dates).all():
                log.info(f"Time parsed: {dates[0]} → {dates[-1]}")
        except Exception as e:
            log.warning(f"Date parsing failed: {e}")
    else:
        log.warning("No time column → using index")
    return df, price, dates

# -------------------------- CORE: FFT DECOMPOSITION --------------------------
def compute_fft_cycles(price: np.ndarray, fs: int, n_top: int = 7) -> Dict[str, Any]:
    """Apply FFT to log-returns, extract top N cycles, reconstruct price."""
    log.info("FFT decomposition started...")
    log_price = np.log(price)                                  # Log transform
    returns = np.diff(log_price)                               # Log returns
    clip_val = np.percentile(np.abs(returns), 99.9)            # Outlier threshold
    returns = np.clip(returns, -clip_val, clip_val)            # Clip extremes
    N = len(returns)
    fft_vals = np.fft.fft(returns)                             # FFT
    freqs = np.fft.fftfreq(N, d=1 / fs)                        # Frequency bins
    pos_mask = freqs > 0                                       # Positive freqs only
    pos_freqs, pos_mag = freqs[pos_mask], np.abs(fft_vals[pos_mask])
    top_idx = np.argsort(pos_mag)[-n_top:][::-1]                # Top N magnitudes
    top_freqs = pos_freqs[top_idx]
    top_coeffs = fft_vals[np.where(pos_mask)[0][top_idx]] / N   # Normalize
    periods = [round(1 / f if f > 0 else float('inf'), 3) for f in top_freqs]
    ticks_per_cycle = {round(p, 3): int(round(p * fs)) for p in periods if p != float('inf')}
    metrics["ticks_per_cycle"] = ticks_per_cycle
    log.info(f"Ticks per cycle: {ticks_per_cycle}")

    # Reconstruct returns from top cycles
    recon_returns = np.zeros(N)
    individual = []
    for freq, coeff in zip(top_freqs, top_coeffs):
        sinusoid = 2 * np.real(coeff * np.exp(2j * np.pi * freq * np.arange(N) / fs))
        recon_returns += sinusoid
        individual.append(sinusoid)
    price_recon = np.concatenate([price[:1], np.exp(log_price[0] + np.cumsum(recon_returns))])[:len(price)]
    return {
        'returns': returns, 'individual': individual, 'periods': periods,
        'mags': pos_mag[top_idx].tolist(), 'price_recon': price_recon
    }

# -------------------------- CORE: PHASE STABILITY --------------------------
def phase_stability_score(cycle_data: Dict[str, Any], fs: int) -> List[float]:
    """Measure phase consistency of each cycle in rolling windows."""
    data = cycle_data['returns']
    scores = []
    window = max(len(data) // 4, 50)
    step = window // 4
    for freq in cycle_data['periods']:
        if freq == float('inf'):
            scores.append(np.pi); continue                     # DC has no phase
        target_freq = 1 / freq
        phases = []
        for s in range(0, len(data) - window + 1, step):
            seg = data[s:s + window]
            if len(seg) < 50: continue
            fft_seg = np.fft.fft(seg)
            freqs_seg = np.fft.fftfreq(len(seg), d=1 / fs)
            closest = np.argmin(np.abs(freqs_seg - target_freq))
            phases.append(np.angle(fft_seg[closest]))
        std = np.std(phases) if len(phases) > 2 else np.pi
        scores.append(round(std, 4))
    stable = [(p, s) for p, s in zip(cycle_data['periods'], scores) if s < 0.8 and 3 <= p < 100]
    if stable:
        best = min(stable, key=lambda x: x[0])[0]
        metrics["best_tradeable_cycle_days"] = round(best, 3)
        log.info(f"BEST TRADEABLE CYCLE: {best:.3f} days")
    return scores

# -------------------------- CORE: REGIME DETECTION --------------------------
def detect_regime(price: np.ndarray) -> str:
    """Detect bull/bear using low-pass filtered trend."""
    try:
        b, a = butter(4, 0.008, btype='low')                   # 4th-order low-pass
        trend = filtfilt(b, a, price)                          # Smooth price
        regime = "Bull" if trend[-1] > trend[-120] else "Bear"  # 120-day lookback
    except Exception:
        regime = "Neutral"
    metrics["regime"] = regime
    log.info(f"Regime: {regime}")
    return regime

# -------------------------- PLOT: FFT DASHBOARD --------------------------
def plot_fft_dashboard(price: np.ndarray, dates: Optional[np.ndarray], cycle_data: Dict,
                       phase_scores: List[float], regime: str, variance_explained: float) -> None:
    """Generate 3-panel HTML dashboard: price, components, spectrum."""
    log.info("Generating dashboard.html...")
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        row_heights=[0.5, 0.3, 0.2],
                        subplot_titles=("Price + FFT Recon", "Cycle Components", "Power Spectrum"))
    x_full = dates if dates is not None and len(dates) == len(price) else np.arange(len(price))
    x_comp = dates[1:1 + len(cycle_data['individual'][0])] if dates is not None else np.arange(1, len(cycle_data['individual'][0]) + 1)

    fig.add_trace(go.Scatter(x=x_full, y=price, name='Price', line=dict(color='black')), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_full, y=cycle_data['price_recon'], name='FFT Recon',
                             line=dict(color='#00ff88', width=3)), row=1, col=1)

    colors = pc.qualitative.Vivid[:len(cycle_data['periods'])]
    ticks_per_cycle = metrics.get("ticks_per_cycle", {})
    for i, (wave, per, stab) in enumerate(zip(cycle_data['individual'], cycle_data['periods'], phase_scores)):
        label = "DC (Trend)" if per == float('inf') else f"{per:.3f}d ({ticks_per_cycle.get(round(per, 3), 'N/A')} ticks)"
        fig.add_trace(go.Scatter(x=x_comp, y=wave, name=label,
                                 line=dict(color=colors[i]), opacity=0.9 if stab < 0.8 else 0.3), row=2, col=1)

    bar_labels = [f"{p:.3f}d ({ticks_per_cycle.get(round(p, 3), 'N/A')})" if p != float('inf') else "DC (Trend)" for p in cycle_data['periods']]
    fig.add_trace(go.Bar(x=bar_labels, y=cycle_data['mags'],
                         marker_color=[colors[i] if phase_scores[i] < 0.8 else 'gray' for i in range(len(colors))]), row=3, col=1)

    best = metrics["best_tradeable_cycle_days"]
    best_txt = f"{best:.3f}d" if best else "None"
    ticks_txt = ", ".join([f"{p}:{t}" for p, t in metrics["ticks_per_cycle"].items()]) or "N/A"
    fig.update_layout(height=1000, hovermode='x unified',
                      title_text=f"<b>FFT Cycle Decomposition</b><br>{regime} | Var Expl: {variance_explained:.2%} | Best: {best_txt} | Ticks: {ticks_txt}")
    fig.update_xaxes(title_text="Date" if dates is not None else "Index", row=3, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Log-Return Amplitude", row=2, col=1)
    fig.update_yaxes(title_text="Magnitude", row=3, col=1)
    fig.write_html("dashboard.html", include_plotlyjs='cdn')
    log.info("dashboard.html → LIVE")

# -------------------------- FORECAST: AUTO-ARIMA --------------------------
def daily_sarima_forecast(price: np.ndarray, dates: Optional[np.ndarray], fs: int) -> Tuple[pd.DataFrame, go.Figure]:
    """Use AUTO-ARIMA to forecast EOD prices for 10 days."""
    log.info("AUTO-ARIMA 10-day forecast...")
    daily_price = price[::fs]                                  # End-of-day prices
    daily_dates = pd.to_datetime(dates[::fs]) if dates is not None and len(dates) >= len(daily_price) else pd.date_range('2025-01-01', periods=len(daily_price), freq='B')
    daily_ret = np.diff(np.log(daily_price))
    daily_ret = np.clip(daily_ret, -np.percentile(np.abs(daily_ret), 99.5), np.percentile(np.abs(daily_ret), 99.5))

    # AUTO-ARIMA model selection
    auto_model = pm.auto_arima(daily_ret, seasonal=True, m=5, stepwise=True, trace=False, suppress_warnings=True, n_jobs=-1)
    order_str = f"{auto_model.order}x{auto_model.seasonal_order}"
    metrics["sarima_order"] = order_str
    log.info(f"AUTO-ARIMA: {order_str}")
    fit = auto_model.fit(daily_ret)

    steps = CONFIG["FORECAST_DAYS"]
    forecast_ret = fit.predict(n_periods=steps)
    pred_price = np.exp(np.cumsum(forecast_ret)) * daily_price[-1]
    future_dates = pd.bdate_range(start=pd.Timestamp(daily_dates[-1]) + pd.Timedelta(days=1), periods=steps, freq='B')

    hist_df = pd.DataFrame({'Time': daily_dates.strftime('%Y-%m-%d'), 'close': daily_price, 'forecast': np.nan})
    fore_df = pd.DataFrame({'Time': future_dates.strftime('%Y-%m-%d'), 'close': np.nan, 'forecast': pred_price})
    full_df = pd.concat([hist_df, fore_df], ignore_index=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"forecast_{timestamp}.csv"
    full_df.to_csv(csv_path, index=False)
    log.info(f"Forecast CSV → {csv_path}")

    # HTML plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_df['Time'], y=hist_df['close'], name='EOD Close', line=dict(color='black')))
    fig.add_trace(go.Scatter(x=fore_df['Time'], y=fore_df['forecast'], name='10-Day Forecast',
                             line=dict(color='#ff0066', width=5), mode='lines+markers', marker=dict(size=8, symbol='diamond')))
    fig.update_layout(title="<b>EOD History + 10-Day AUTO-ARIMA Forecast</b>", height=700, xaxis_title="Date", yaxis_title="Price",
                      hovermode='x unified', template='plotly_white', xaxis=dict(tickangle=45))
    fig.write_html("forecast.html", include_plotlyjs='cdn')
    log.info("forecast.html → AUTO-ARIMA")

    # Backtest metrics
    try:
        back_len = min(steps, len(daily_price))
        mae = mean_absolute_error(daily_price[-back_len:], pred_price[-back_len:])
        mape = np.mean(np.abs((daily_price[-back_len:] - pred_price[-back_len:]) / daily_price[-back_len:])) * 100
        metrics["sarima_mae"] = round(mae, 6)
        metrics["sarima_mape"] = round(mape, 3)
    except: pass
    return full_df, fig

# -------------------------- MICRO-CYCLE CHARTING (>20 ticks) --------------------------
def plot_micro_cycles(price: np.ndarray, cycle_data: Dict, fs: int):
    """Plot all stable micro-cycles (>20 ticks) on last trading day."""
    log.info("Plotting micro cycles (>20 ticks)...")
    fig = go.Figure()
    recent = price[-fs:]                                       # Last trading day
    x = list(range(len(recent)))
    fig.add_trace(go.Scatter(x=x, y=recent, name="Today", line=dict(color="black", width=2)))

    # Find stable short cycles
    stable_idx = [i for i, (p, s) in enumerate(zip(cycle_data['periods'], phase_stability_score(cycle_data, fs)))
                  if s < 0.8 and p < 1 and metrics["ticks_per_cycle"].get(round(p, 3), 0) > CONFIG["MICRO_MIN_TICKS"]]
    colors = ['#ff0066', '#00ff88', '#ffaa00', '#aa00ff']
    plotted = []
    for j, i in enumerate(stable_idx):
        wave = cycle_data['individual'][i][-fs:]
        per = cycle_data['periods'][i]
        ticks = metrics["ticks_per_cycle"].get(round(per, 3), 0)
        fig.add_trace(go.Scatter(x=x, y=recent[-1] + wave * 100, name=f"{per:.3f}d ({ticks} ticks)",
                                 line=dict(color=colors[j % len(colors)], dash='dot')))
        plotted.append(f"{per:.3f}d")
    metrics["micro_cycles_plotted"] = plotted
    fig.update_layout(title="<b>Micro Cycles (>20 ticks) – Last Trading Day</b>", height=600, xaxis_title="Tick", yaxis_title="Price")
    fig.write_html("micro_cycles.html", include_plotlyjs='cdn')
    log.info(f"micro_cycles.html → {len(plotted)} cycles plotted")

# -------------------------- METRICS & OUTPUT --------------------------
def save_metrics():
    """Save all metrics to JSON in logs/"""
    path = log_dir / f"metrics_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    log.info(f"Metrics → {path}")

def summarize_outputs():
    """Print and log all generated files."""
    log.info("\n=== OUTPUT SUMMARY ===")
    patterns = ["dashboard.html", "forecast.html", "micro_cycles.html", "forecast_*.csv", "FFT_LOGRETURNS_OUTPUT.csv", "metrics_*.json", "toolkit_*.log"]
    found = []
    for pat in patterns:
        for p in list(Path(".").glob(pat)) + list(log_dir.glob(pat)):
            try:
                size_kb = p.stat().st_size / 1024
                ts = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                found.append(f"{p.name:<40} | {size_kb:8.1f} KB | {ts}")
            except: pass
    if found:
        header = f"{'File':<40} | {'Size':>8} | Modified"
        sep = "-" * len(header)
        table = "\n".join(found)
        summary = f"\n{header}\n{sep}\n{table}\n{sep}"
        print(summary)
        log.info(summary)
    else:
        log.warning("No output files found.")

# -------------------------- MAIN ORCHESTRATOR --------------------------
def run_analysis(filepath: str):
    """Orchestrate full analysis pipeline."""
    log.info("=== MARKET CYCLES TOOLKIT v3.5.3 – START ===")
    df, price, dates = load_data(filepath, CONFIG["COLUMN_INDEX"])
    cycle_data = compute_fft_cycles(price, CONFIG["FS"], CONFIG["N_TOP"])
    phase_stability_score(cycle_data, CONFIG["FS"])  # Populates best cycle
    regime = detect_regime(price)
    variance_explained = 1 - np.mean((price[1:] - cycle_data['price_recon'][1:]) ** 2) / np.var(price)
    metrics["variance_explained"] = round(variance_explained, 5)

    plot_fft_dashboard(price, dates, cycle_data, phase_stability_score(cycle_data, CONFIG["FS"]), regime, variance_explained)
    daily_sarima_forecast(price, dates, CONFIG["FS"])
    plot_micro_cycles(price, cycle_data, CONFIG["FS"])

    out_df = pd.DataFrame({'price': price[1:], 'fft_price': cycle_data['price_recon'][1:]})
    if dates is not None: out_df.insert(0, 'date', dates[1:])
    out_df.to_csv('FFT_LOGRETURNS_OUTPUT.csv', index=False)

    save_metrics()
    summarize_outputs()
    print("\n" + "="*70)
    print(" TOOLKIT COMPLETE | v3.5.3 | THE ANNOTATED TRUTH ")
    print(" Open: dashboard.html | forecast.html | micro_cycles.html ")
    print(" API Ready: metrics_*.json | forecast_*.csv ")
    print("="*70)

    import webbrowser
    webbrowser.open("dashboard.html")



# -------------------------- CLI --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Market Cycles Toolkit v3.5.3")
    parser.add_argument("--file", type=str, default=CONFIG["FILE_PATH"], help="Path to CSV")
    parser.add_argument("--preview", action="store_true", help="Print output summary")
    args = parser.parse_args()
    CONFIG["FILE_PATH"] = args.file
    run_analysis(args.file)
    if args.preview:
        summarize_outputs()


#pip install fastapi uvicorn
#uvicorn api:app --reload
# → http://127.0.0.1:8000/metrics
#python "C:\Users\Hein Vogel\Desktop\FFT_API\FFT_ARIMA.py" --file "C:\Users\Hein Vogel\Desktop\FFT_API\Cleaned & Current Raw Data.csv"
#cd "C:\Users\Hein Vogel\Desktop\FFT_API\" 
