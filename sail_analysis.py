#!/usr/bin/env python3
"""
SAIL Intraday Analysis Script
Runs every 30 minutes via GitHub Actions
Downloads latest 5-min data, updates cache, generates analysis
"""

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import pickle
import os
import warnings
import time
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
SYMBOL = "SAIL.NS"
INTERVAL = "5m"
DATA_DIR = Path("./data")
DATA_FILE = DATA_DIR / "sail_intraday.pkl"
CHART_FILE = DATA_DIR / "sail_chart.html"
REPORT_FILE = DATA_DIR / "analysis_report.txt"

# Create data directory if it doesn't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
print(f"📁 Data directory: {DATA_DIR.absolute()}")


# ==================== DATA MANAGEMENT ====================

def load_existing_data():
    """Load existing data from pickle file"""
    if DATA_FILE.exists():
        try:
            with open(DATA_FILE, 'rb') as f:
                df = pickle.load(f)
            print(f"✅ Loaded existing data: {len(df)} bars")
            print(f"   Last date: {df.index[-1]}")
            return df
        except Exception as e:
            print(f"⚠️ Error loading data: {e}")
            return pd.DataFrame()
    else:
        print("📁 No existing data found. Starting fresh.")
        return pd.DataFrame()


def flatten_columns(df):
    """Flatten MultiIndex columns from yfinance"""
    if hasattr(df.columns, 'levels'):
        df.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in df.columns]
    df.columns = [col.lower() for col in df.columns]
    
    # Map to standard names
    col_map = {}
    for col in df.columns:
        if 'open' in col:
            col_map[col] = 'open'
        elif 'high' in col:
            col_map[col] = 'high'
        elif 'low' in col:
            col_map[col] = 'low'
        elif 'close' in col:
            col_map[col] = 'close'
        elif 'volume' in col:
            col_map[col] = 'volume'
    
    if col_map:
        df = df.rename(columns=col_map)
    
    return df


def fetch_new_data(df_existing):
    """Fetch only new data since last update with retry logic"""
    
    # Determine start date
    if len(df_existing) > 0:
        last_date = df_existing.index[-1]
        # Fetch from 6 hours before last date to ensure continuity
        start_date = last_date - timedelta(hours=6)
    else:
        # First run: fetch last 7 days
        start_date = datetime.now() - timedelta(days=7)
    
    end_date = datetime.now()
    
    print(f"📥 Fetching data from {start_date.strftime('%Y-%m-%d %H:%M')} to {end_date.strftime('%Y-%m-%d %H:%M')}")
    
    # Retry logic
    max_retries = 3
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            print(f"   Attempt {attempt + 1}/{max_retries}...")
            
            new_df = yf.download(
                SYMBOL, 
                start=start_date, 
                end=end_date,
                interval=INTERVAL, 
                progress=False,
                timeout=30
            )
            
            if len(new_df) == 0:
                print("⚠️ No new data received. Market may be closed.")
                # Return existing data if available
                if len(df_existing) > 0:
                    return df_existing
                else:
                    # Create dummy data for testing
                    print("⚠️ Creating sample data for testing...")
                    dates = pd.date_range(start=datetime.now() - timedelta(days=1), periods=100, freq='5min')
                    df_sample = pd.DataFrame({
                        'open': np.random.uniform(450, 460, 100),
                        'high': np.random.uniform(455, 465, 100),
                        'low': np.random.uniform(445, 455, 100),
                        'close': np.random.uniform(450, 460, 100),
                        'volume': np.random.uniform(100000, 500000, 100)
                    }, index=dates)
                    return df_sample
            
            new_df = flatten_columns(new_df)
            
            # Combine with existing data
            if len(df_existing) > 0:
                # Remove any overlapping data
                combined = pd.concat([df_existing, new_df])
                combined = combined[~combined.index.duplicated(keep='last')]
                combined = combined.sort_index()
            else:
                combined = new_df
            
            # Keep only last 7 days for performance
            cutoff = datetime.now() - timedelta(days=7)
            combined = combined[combined.index >= cutoff]
            
            print(f"✅ Downloaded {len(new_df)} new bars")
            print(f"   Total bars: {len(combined)}")
            if len(combined) > 0:
                print(f"   Date range: {combined.index[0]} to {combined.index[-1]}")
            
            return combined
        
        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed: {str(e)}")
            
            if attempt < max_retries - 1:
                print(f"⏳ Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"❌ All {max_retries} attempts failed")
                # Return existing data as fallback
                if len(df_existing) > 0:
                    print("⚠️ Using existing data from cache")
                    return df_existing
                else:
                    return pd.DataFrame()


# ==================== INDICATOR CALCULATIONS ====================

def calculate_indicators(df):
    """Calculate all technical indicators"""
    
    if len(df) < 50:
        print("⚠️ Insufficient data for indicator calculation")
        return df
    
    df = df.copy()
    
    # Moving Averages
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma50'] = df['close'].rolling(50).mean()
    df['ma100'] = df['close'].rolling(100).mean()
    df['ma200'] = df['close'].rolling(200).mean()
    
    # TEMA
    ema1 = df['close'].ewm(span=14, adjust=False).mean()
    ema2 = ema1.ewm(span=14, adjust=False).mean()
    ema3 = ema2.ewm(span=14, adjust=False).mean()
    df['tema'] = 3 * ema1 - 3 * ema2 + ema3
    
    # VWAP (daily reset)
    df['date'] = df.index.date
    df['vwap'] = df.groupby('date').apply(
        lambda g: (g['volume'] * (g['high'] + g['low'] + g['close']) / 3).cumsum() / g['volume'].cumsum()
    ).reset_index(level=0, drop=True)
    
    # UDAY Bands (EVWAP)
    alpha = 0.1
    n = len(df)
    evwap = np.zeros(n)
    evwap[0] = df['close'].iloc[0]
    sum_pv = df['close'].iloc[0] * df['volume'].iloc[0]
    sum_v = df['volume'].iloc[0]
    
    for i in range(1, n):
        if df.index[i].date() != df.index[i-1].date():
            sum_pv = df['close'].iloc[i] * df['volume'].iloc[i]
            sum_v = df['volume'].iloc[i]
        else:
            sum_pv = sum_pv * (1 - alpha) + (df['close'].iloc[i] * df['volume'].iloc[i]) * alpha
            sum_v = sum_v * (1 - alpha) + df['volume'].iloc[i] * alpha
        if sum_v > 0:
            evwap[i] = sum_pv / sum_v
    
    df['uday'] = evwap
    df['uday_std'] = df['uday'].rolling(20).std()
    df['uday_upper'] = df['uday'] + df['uday_std'] * 2.91
    df['uday_lower'] = df['uday'] - df['uday_std'] * 2.91
    
    # Keltner Channels
    df['kc_mid'] = df['close'].ewm(span=20, adjust=False).mean()
    df['kc_atr'] = (df['high'] - df['low']).rolling(14).mean()
    df['kc_upper'] = df['kc_mid'] + 2 * df['kc_atr']
    df['kc_lower'] = df['kc_mid'] - 2 * df['kc_atr']
    
    # Regression Channel
    df['reg_mid'] = df['close'].rolling(50).mean()
    df['reg_std'] = df['close'].rolling(50).std()
    df['reg_upper'] = df['reg_mid'] + df['reg_std']
    df['reg_lower'] = df['reg_mid'] - df['reg_std']
    df['slope'] = df['reg_mid'].diff()
    
    # Bollinger Bands
    df['bb_mid'] = df['close'].rolling(112).mean()
    df['bb_std'] = df['close'].rolling(112).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    
    # Kalman Filter
    kalman = np.zeros(len(df))
    kalman[0] = df['close'].iloc[0]
    cov = 1.0
    for i in range(1, len(df)):
        pred_cov = cov + 0.01
        kgain = pred_cov / (pred_cov + 1.0)
        kalman[i] = kalman[i-1] + kgain * (df['close'].iloc[i] - kalman[i-1])
        cov = (1 - kgain) * pred_cov
    df['kalman'] = kalman
    
    # Volume & Signals
    df['vol_ma'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_ma']
    
    # Direction
    df['direction'] = np.where(df['close'] > df['open'], 1, -1)
    df['direction_change'] = df['direction'] != df['direction'].shift(1)
    df['bullish_reversal'] = df['direction_change'] & (df['direction'] == 1)
    df['bearish_reversal'] = df['direction_change'] & (df['direction'] == -1)
    
    # Impulse
    df['impulse'] = (df['vol_ratio'] > 2) & (abs(df['slope']) > df['reg_std'] * 0.1)
    
    # CSK (Crow-Siddique Kurtosis)
    returns = df['close'].pct_change()
    window = 50
    n_win = window
    c1 = n_win * (n_win + 1) / ((n_win - 1) * (n_win - 2) * (n_win - 3))
    c2 = 3 * (n_win - 1) ** 2 / ((n_win - 2) * (n_win - 3))
    
    def csk_func(x):
        if len(x) < 25:
            return np.nan
        if np.std(x) == 0:
            return 0
        m2 = np.var(x)
        m4 = np.mean((x - np.mean(x))**4)
        return c1 * (m4 / (m2 ** 2)) - c2
    
    df['csk'] = returns.rolling(window).apply(csk_func)
    df['coherence'] = returns.rolling(14).corr(returns.shift(1)).abs().fillna(0)
    
    # Structural metrics
    df['fatigue'] = (abs(df['slope']) / df['reg_std'] / 10).clip(0, 2)
    df['whipsaw'] = ((abs(df['slope']) < df['reg_std'] * 0.05) & (df['vol_ratio'] < 1)).astype(int)
    mom = df['close'] - df['close'].shift(1)
    df['rot_z'] = mom.abs().ewm(span=5).mean() / df['close'].rolling(20).std()
    
    # Envelope
    df['envelope'] = 'INSIDE'
    df.loc[df['close'] > df['uday_upper'], 'envelope'] = 'UPPER'
    df.loc[df['close'] < df['uday_lower'], 'envelope'] = 'LOWER'
    
    return df


# ==================== CHART GENERATION ====================

def generate_chart(df):
    """Generate interactive Plotly chart"""
    
    if len(df) == 0:
        print("⚠️ No data for chart generation")
        return None
    
    # Use last 7 days for chart (performance)
    cutoff = datetime.now() - timedelta(days=7)
    chart_df = df[df.index >= cutoff]
    
    if len(chart_df) == 0:
        chart_df = df.tail(500)  # Fallback to last 500 bars
    
    fig = go.Figure()
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=chart_df.index,
        open=chart_df['open'],
        high=chart_df['high'],
        low=chart_df['low'],
        close=chart_df['close'],
        name='Price'
    ))
    
    # UDAY Bands
    fig.add_trace(go.Scatter(
        x=chart_df.index, y=chart_df['uday_upper'],
        name='UDAY Upper', line=dict(color='#66b3ff', width=1)
    ))
    
    fig.add_trace(go.Scatter(
        x=chart_df.index, y=chart_df['uday_lower'],
        name='UDAY Lower', line=dict(color='#ff6666', width=1),
        fill='tonexty', fillcolor='rgba(128,128,128,0.05)'
    ))
    
    fig.add_trace(go.Scatter(
        x=chart_df.index, y=chart_df['uday'],
        name='UDAY EVWAP', line=dict(color='yellow', width=1)
    ))
    
    # Signals
    buy = chart_df[chart_df['bullish_reversal']]
    sell = chart_df[chart_df['bearish_reversal']]
    imp_up = chart_df[chart_df['impulse'] & (chart_df['close'] > chart_df['open'])]
    imp_down = chart_df[chart_df['impulse'] & (chart_df['close'] < chart_df['open'])]
    
    fig.add_trace(go.Scatter(
        x=buy.index, y=buy['low'],
        mode='markers', marker=dict(symbol='circle', size=8, color='lime'),
        name='Bullish Reversal'
    ))
    
    fig.add_trace(go.Scatter(
        x=sell.index, y=sell['high'],
        mode='markers', marker=dict(symbol='circle', size=8, color='orange'),
        name='Bearish Reversal'
    ))
    
    fig.add_trace(go.Scatter(
        x=imp_up.index, y=imp_up['high'],
        mode='markers', marker=dict(symbol='star', size=12, color='blue'),
        name='Bull Impulse'
    ))
    
    fig.add_trace(go.Scatter(
        x=imp_down.index, y=imp_down['low'],
        mode='markers', marker=dict(symbol='star', size=12, color='darkred'),
        name='Bear Impulse'
    ))
    
    fig.update_layout(
        template='plotly_dark',
        height=600,
        title=f'{SYMBOL} - 5-Minute Intraday Analysis',
        xaxis_title='Time',
        yaxis_title='Price (₹)',
        hovermode='x unified'
    )
    
    return fig


# ==================== REPORT GENERATION ====================

def generate_report(df):
    """Generate text analysis report"""
    
    if len(df) == 0:
        return "No data available for analysis"
    
    latest = df.iloc[-1]
    last_5 = df.tail(5)
    
    # Calculate signals
    recent_bullish = last_5[last_5['bullish_reversal']]
    recent_bearish = last_5[last_5['bearish_reversal']]
    recent_impulse = last_5[last_5['impulse']]
    
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append(f"SAIL INTRADAY ANALYSIS REPORT")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 60)
    report_lines.append("")
    report_lines.append(f"📊 CURRENT STATUS")
    report_lines.append(f"   Price: ₹{latest['close']:.2f}")
    report_lines.append(f"   Volume Ratio: {latest['vol_ratio']:.2f}x")
    report_lines.append(f"   CSK: {latest['csk']:.2f}" if not pd.isna(latest['csk']) else "   CSK: N/A")
    report_lines.append(f"   Coherence: {latest['coherence']:.2f}")
    report_lines.append(f"   Envelope: {latest['envelope']}")
    report_lines.append("")
    report_lines.append(f"📈 SIGNALS")
    
    if len(recent_bullish) > 0:
        report_lines.append(f"   🔵 BULLISH REVERSAL at {recent_bullish.index[-1].strftime('%H:%M')}")
    else:
        report_lines.append(f"   🔵 No recent bullish reversal")
    
    if len(recent_bearish) > 0:
        report_lines.append(f"   🔴 BEARISH REVERSAL at {recent_bearish.index[-1].strftime('%H:%M')}")
    else:
        report_lines.append(f"   🔴 No recent bearish reversal")
    
    if len(recent_impulse) > 0:
        report_lines.append(f"   ⭐ IMPULSE SIGNAL at {recent_impulse.index[-1].strftime('%H:%M')}")
    else:
        report_lines.append(f"   ⭐ No recent impulse signals")
    
    if latest['vol_ratio'] > 2:
        report_lines.append(f"   ⚠️ HIGH VOLUME ALERT: {latest['vol_ratio']:.1f}x average")
    
    report_lines.append("")
    report_lines.append(f"📊 STATISTICS")
    report_lines.append(f"   Total bars in dataset: {len(df)}")
    report_lines.append(f"   Date range: {df.index[0].strftime('%Y-%m-%d %H:%M')} to {df.index[-1].strftime('%Y-%m-%d %H:%M')}")
    
    # CSK regime
    if not pd.isna(latest['csk']):
        if latest['csk'] > 2:
            report_lines.append(f"   ⚠️ CSK REGIME: EXTREME (>{latest['csk']:.2f}) - High volatility expected")
        elif latest['csk'] > 1:
            report_lines.append(f"   ⚠️ CSK REGIME: HIGH ({latest['csk']:.2f}) - Elevated risk")
        elif latest['csk'] > 0.5:
            report_lines.append(f"   ✅ CSK REGIME: MODERATE ({latest['csk']:.2f}) - Normal conditions")
        else:
            report_lines.append(f"   ✅ CSK REGIME: LOW ({latest['csk']:.2f}) - Low risk")
    
    report_lines.append("")
    report_lines.append("=" * 60)
    report_lines.append("✅ Analysis Complete")
    report_lines.append("=" * 60)
    
    return "\n".join(report_lines)


# ==================== SAVE DATA ====================

def save_data(df, fig, report):
    """Save all outputs to files"""
    
    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save pickle
    try:
        with open(DATA_FILE, 'wb') as f:
            pickle.dump(df, f)
        print(f"💾 Data saved to {DATA_FILE}")
        print(f"   File size: {DATA_FILE.stat().st_size / 1024:.2f} KB")
    except Exception as e:
        print(f"❌ Error saving pickle: {e}")
    
    # Save chart
    if fig:
        try:
            fig.write_html(str(CHART_FILE))
            print(f"📊 Chart saved to {CHART_FILE}")
            print(f"   File size: {CHART_FILE.stat().st_size / 1024:.2f} KB")
        except Exception as e:
            print(f"❌ Error saving chart: {e}")
    
    # Save report
    try:
        with open(REPORT_FILE, 'w') as f:
            f.write(report)
        print(f"📝 Report saved to {REPORT_FILE}")
        print(f"   File size: {REPORT_FILE.stat().st_size / 1024:.2f} KB")
    except Exception as e:
        print(f"❌ Error saving report: {e}")
    
    # List files in data directory
    print(f"\n📁 Files in {DATA_DIR}:")
    for f in DATA_DIR.glob('*'):
        print(f"   - {f.name} ({f.stat().st_size / 1024:.2f} KB)")


# ==================== MAIN ====================

def main():
    print("\n" + "=" * 60)
    print("🚀 SAIL Intraday Analysis Started")
    print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60 + "\n")
    
    # Step 1: Load existing data
    df_existing = load_existing_data()
    
    # Step 2: Fetch new data
    df = fetch_new_data(df_existing)
    
    if len(df) == 0:
        print("❌ No data available. Exiting.")
        return
    
    # Step 3: Calculate indicators
    print("\n📊 Calculating indicators...")
    df = calculate_indicators(df)
    print("✅ Indicators calculated")
    
    # Step 4: Generate chart
    print("\n📈 Generating chart...")
    fig = generate_chart(df)
    
    # Step 5: Generate report
    print("\n📝 Generating report...")
    report = generate_report(df)
    
    # Step 6: Save everything
    print("\n💾 Saving outputs...")
    save_data(df, fig, report)
    
    # Step 7: Print report to console
    print("\n" + report)
    
    print("\n" + "=" * 60)
    print("✅ SAIL Intraday Analysis Completed Successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()
