# -*- coding: utf-8 -*-
"""
AYAR ENHANCED - سیستم معاملاتی پیشرفته بورس ایران
================================================================
بهبودها:
- Market filter (صف/رنج/حجم مجاز)
- Adaptive parameters (ATR-based stops)
- Correlation با شاخص کل
- Advanced error handling + retry
- Logging system
- Performance metrics
- Backtest foundation
- Multi-symbol support
"""

import argparse
import json
import logging
import random
import re
import time
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import requests


# =========================
# CONFIGURATION
# =========================
@dataclass
class Config:
    # Capital & Risk
    capital_rial: int = 500_000_000
    risk_pct: float = 1.0
    max_exposure_pct: float = 95.0
    max_price_for_qty: int = 5_000_000
    
    # Entry
    min_real_power: float = 1.5
    vol_mult: float = 2.0
    val_mult: float = 2.0
    min_score: int = 3
    allow_negative_entry: bool = False  # اجازه ورود در قیمت منفی (برای کف‌شکن‌ها)
    
    # Depth
    tight_spread_pct: float = 0.15
    depth_imb_min: float = 1.30
    ask_wall_dom_ratio: float = 0.60
    top_pressure_alert: float = 3.0
    
    # Exit
    stop_loss_pct: float = 1.20
    take_profit_pct: float = 1.80
    trail_after_profit_pct: float = 1.00
    rp_exit_level: float = 1.10
    
    # Market Filter
    enable_market_filter: bool = True
    max_queue_ratio: float = 0.70  # حداکثر نسبت صف خرید
    min_price_change_pct: float = -2.0  # حداقل تغییر قیمت
    max_price_change_pct: float = 4.9  # حداکثر تغییر (زیر صف)
    
    # Correlation
    enable_index_correlation: bool = True
    min_index_change_pct: float = -0.5  # شاخص نباید خیلی منفی باشد
    
    # ATR (Adaptive stops)
    use_atr_stops: bool = True
    atr_period: int = 14
    atr_stop_multiplier: float = 2.0
    
    # System
    timeout: int = 20
    max_retries: int = 3
    retry_delay: float = 1.0


STATE_FILE = Path("ayar_state.json")
LOG_FILE = Path("ayar_trading.log")
METRICS_FILE = Path("ayar_metrics.json")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# =========================
# CONSTANTS
# =========================
BASES = [
    "https://www.tsetmc.com",
    "https://cdn.tsetmc.com",
    "https://service.tsetmc.com",
]

UAS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Mozilla/5.0 (Linux; Android 13) AppleWebKit/537.36 Mobile",
]

HEADERS = {
    "Accept": "*/*",
    "Accept-Language": "fa-IR,fa;q=0.9",
    "Connection": "keep-alive",
    "Cache-Control": "no-cache",
}

_num_token_re = re.compile(r"^-?\d+(\.\d+)?$")


# =========================
# UTILITIES
# =========================
def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_int(x, default=0) -> int:
    try:
        s = str(x).strip().replace(",", "")
        if s == "" or s.lower() == "nan":
            return default
        if "." in s:
            s = s.split(".")[0]
        return int(s)
    except Exception:
        return default


def safe_float(x, default=None) -> Optional[float]:
    try:
        s = str(x).strip().replace(",", "")
        if s == "" or s.lower() == "nan":
            return default
        return float(s)
    except Exception:
        return default


def fmt(x: Optional[float], nd: int = 2) -> str:
    if x is None:
        return "NA"
    if abs(x) >= 1000 and float(int(x)) == x:
        return f"{int(x):,}"
    return f"{x:.{nd}f}"


# =========================
# HTTP WITH RETRY
# =========================
def http_get_with_retry(
    path: str,
    params: Optional[dict] = None,
    min_len: int = 50,
    timeout: int = 20,
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> str:
    last_err = None
    sess = requests.Session()
    
    for attempt in range(max_retries):
        bases = BASES[:]
        random.shuffle(bases)
        
        for base in bases:
            url = base.rstrip("/") + path
            try:
                h = dict(HEADERS)
                h["User-Agent"] = random.choice(UAS)
                r = sess.get(url, headers=h, params=params, timeout=timeout)
                txt = (r.text or "").strip()
                
                if r.status_code != 200:
                    last_err = f"HTTP {r.status_code}"
                    continue
                
                if len(txt) < min_len:
                    last_err = f"short response {len(txt)}"
                    continue
                
                head = txt[:250].lower()
                if ("<html" in head) or ("<!doctype" in head):
                    last_err = "HTML/blocked"
                    continue
                
                return txt
                
            except requests.Timeout:
                last_err = "timeout"
            except requests.RequestException as e:
                last_err = str(e)
            except Exception as e:
                last_err = str(e)
            
            time.sleep(0.1 + random.random() * 0.2)
        
        if attempt < max_retries - 1:
            time.sleep(retry_delay * (attempt + 1))
    
    raise RuntimeError(f"fetch failed after {max_retries} attempts: {path}. last={last_err}")


# =========================
# DATA MODELS
# =========================
@dataclass
class DepthLevel:
    level: int
    bid_price: int
    bid_qty: int
    bid_orders: int
    ask_price: int
    ask_qty: int
    ask_orders: int


@dataclass
class MarketData:
    ins_code: str
    last_price: int
    close_price: int
    yest_price: int
    vol_today: int
    val_today: int
    low_today: int
    high_today: int
    chg_pct: Optional[float]
    avg_vol20: int
    avg_val5: int
    real_power: float
    net_real_vol: int
    depth: Optional[List[DepthLevel]]
    depth_metrics: Dict[str, Optional[float]]
    index_change_pct: Optional[float]
    atr: Optional[float]


@dataclass
class Signal:
    enter: bool
    exit: bool
    exit_reason: Optional[str]
    score: int
    conditions: Dict[str, bool]
    depth_ok: bool
    market_filter_pass: bool
    correlation_ok: bool


@dataclass
class Position:
    entry_price: int
    peak_price: int
    qty: int
    entry_time: str
    stop_loss: int
    take_profit: int


# =========================
# MARKET DATA FETCHING
# =========================
def parse_marketwatchinit(text: str) -> Tuple[List[List[str]], List[List[str]]]:
    prices_rows: List[List[str]] = []
    best_rows: List[List[str]] = []
    for part in text.split("@"):
        part = part.strip()
        if not part:
            continue
        for row in [r for r in part.split(";") if r.strip()]:
            cols = row.split(",")
            if len(cols) == 8 and cols[0].strip().isdigit():
                best_rows.append(cols)
            elif len(cols) >= 23 and cols[0].strip().isdigit():
                prices_rows.append(cols)
    return prices_rows, best_rows


def get_price_row(prices_rows: List[List[str]], ins_code: str) -> Optional[List[str]]:
    for cols in prices_rows:
        if cols and cols[0] == ins_code:
            return cols
    return None


def extract_depth(best_rows: List[List[str]], ins_code: str, levels: int = 5) -> Optional[List[DepthLevel]]:
    rows = [r for r in best_rows if r and r[0] == ins_code]
    if not rows:
        return None
    rows.sort(key=lambda r: safe_int(r[1], 999))
    
    out: List[DepthLevel] = []
    for r in rows[:levels]:
        out.append(
            DepthLevel(
                level=safe_int(r[1]),
                bid_orders=safe_int(r[2]),
                ask_orders=safe_int(r[3]),
                bid_price=safe_int(r[4]),
                ask_price=safe_int(r[5]),
                bid_qty=safe_int(r[6]),
                ask_qty=safe_int(r[7]),
            )
        )
    return out if out else None


def fetch_financial_avgs(ins_code: str, config: Config) -> Tuple[int, int, Optional[float]]:
    """Returns: (avg_vol20, avg_val5, atr)"""
    try:
        txt = http_get_with_retry(
            "/tsev2/chart/data/Financial.aspx",
            params={"i": ins_code, "t": "ph", "a": 1},
            min_len=200,
            timeout=config.timeout,
            max_retries=config.max_retries
        )
        
        rows: List[Tuple[int, int, int, int]] = []  # (vol, close, high, low)
        for ln in txt.replace(";", "\n").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            cols = [c.strip() for c in ln.split(",")]
            if len(cols) < 5:
                continue
            high = safe_int(cols[2])
            low = safe_int(cols[3])
            vol = safe_int(cols[-2])
            close = safe_int(cols[-1])
            if vol > 0 and close > 0:
                rows.append((vol, close, high, low))
        
        if len(rows) < 5:
            return 0, 0, None
        
        vols = [v for v, _, _, _ in rows]
        avg_vol20 = int(sum(vols[-20:]) / max(1, min(20, len(vols))))
        
        proxy_vals = [v * c for v, c, _, _ in rows]
        avg_val5 = int(sum(proxy_vals[-5:]) / 5)
        
        # Calculate ATR
        atr = None
        if config.use_atr_stops and len(rows) >= config.atr_period:
            trs = []
            for i in range(1, len(rows)):
                _, prev_close, _, _ = rows[i-1]
                _, curr_close, curr_high, curr_low = rows[i]
                tr = max(
                    curr_high - curr_low,
                    abs(curr_high - prev_close),
                    abs(curr_low - prev_close)
                )
                trs.append(tr)
            if trs:
                atr = sum(trs[-config.atr_period:]) / min(config.atr_period, len(trs))
        
        return avg_vol20, avg_val5, atr
        
    except Exception as e:
        logger.error(f"fetch_financial_avgs error: {e}")
        return 0, 0, None


def fetch_clienttype_numbers(ins_code: str, config: Config) -> List[float]:
    try:
        txt = http_get_with_retry(
            "/tsev2/data/clienttype.aspx",
            params={"i": ins_code},
            min_len=20,
            timeout=config.timeout,
            max_retries=config.max_retries
        )
        tokens = re.split(r"[,\s;]+", txt.replace("\r", " ").replace("\n", " "))
        nums: List[float] = []
        for t in tokens:
            t = t.strip()
            if not t:
                continue
            if _num_token_re.match(t):
                v = safe_float(t, default=None)
                if v is not None:
                    nums.append(v)
        return nums
    except Exception as e:
        logger.error(f"fetch_clienttype error: {e}")
        return []


def infer_realpower_and_netreal(nums: List[float]) -> Tuple[float, int]:
    if not nums or len(nums) < 6:
        return 0.0, 0
    
    nums2 = nums[:60]
    count_idx = [i for i, x in enumerate(nums2) if 0 < x <= 10_000_000]
    vol_idx = [i for i, x in enumerate(nums2) if 0 < x <= 10_000_000_000_000]
    
    best = None
    
    for i in count_idx:
        rb_cnt = int(nums2[i])
        if rb_cnt <= 0:
            continue
        for j in count_idx:
            if j == i:
                continue
            rs_cnt = int(nums2[j])
            if rs_cnt <= 0:
                continue
            
            for a in vol_idx:
                rb_vol = int(nums2[a])
                if rb_vol < rb_cnt:
                    continue
                for b in vol_idx:
                    if b == a:
                        continue
                    rs_vol = int(nums2[b])
                    if rs_vol < rs_cnt:
                        continue
                    
                    sell_avg = rs_vol / rs_cnt if rs_vol > 0 else 0.0
                    if sell_avg <= 0:
                        continue
                    buy_avg = rb_vol / rb_cnt
                    rp = buy_avg / sell_avg
                    
                    if not (0.05 <= rp <= 50):
                        continue
                    
                    net = rb_vol - rs_vol
                    
                    score = 0
                    if rp >= 1.0:
                        score += 2
                    if rp >= 1.5:
                        score += 2
                    if rp >= 2.0:
                        score += 1
                    if rb_cnt <= 10000:
                        score += 1
                    if rs_cnt <= 10000:
                        score += 1
                    if abs(net) <= 2_000_000_000:
                        score += 1
                    
                    cand = (score, rp, net)
                    if best is None or cand[0] > best[0]:
                        best = cand
    
    if best is None:
        return 0.0, 0
    
    return float(best[1]), int(best[2])


def get_index_change(config: Config) -> Optional[float]:
    """Fetch TEPIX overall index change %"""
    try:
        # شاخص کل: InsCode = 32097828799138957
        txt = http_get_with_retry(
            "/tsev2/data/MarketWatchInit.aspx",
            params={"h": "0", "r": "0"},
            min_len=200,
            timeout=config.timeout,
            max_retries=2
        )
        prices_rows, _ = parse_marketwatchinit(txt)
        index_row = get_price_row(prices_rows, "32097828799138957")
        
        if index_row:
            last = safe_int(index_row[7])
            yest = safe_int(index_row[13])
            if yest > 0:
                return ((last - yest) / yest) * 100.0
    except Exception as e:
        logger.warning(f"get_index_change error: {e}")
    
    return None


# =========================
# DEPTH ANALYSIS
# =========================
def depth_metrics_advanced(
    depth: Optional[List[DepthLevel]],
    last_price: int,
    config: Config
) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "spread_pct": None,
        "imb_simple": None,
        "imb_weighted": None,
        "top_pressure": None,
        "ask_share1": None,
        "tight_spread": 0.0,
        "demand_dom": 0.0,
        "ask_wall_strong": 0.0,
    }
    
    if not depth or last_price <= 0:
        return out
    
    top = depth[0]
    if top.bid_price > 0 and top.ask_price > 0:
        out["spread_pct"] = ((top.ask_price - top.bid_price) / last_price) * 100.0
    
    bid_sum = sum(d.bid_qty for d in depth if d.bid_qty > 0)
    ask_sum = sum(d.ask_qty for d in depth if d.ask_qty > 0)
    
    out["imb_simple"] = (bid_sum / ask_sum) if ask_sum > 0 else (10.0 if bid_sum > 0 else 0.0)
    
    weights = [5, 4, 3, 2, 1]
    wb = 0.0
    wa = 0.0
    for i, d in enumerate(depth[:5]):
        w = weights[i]
        wb += w * max(0, d.bid_qty)
        wa += w * max(0, d.ask_qty)
    out["imb_weighted"] = (wb / wa) if wa > 0 else (10.0 if wb > 0 else 0.0)
    
    if top.ask_qty > 0 or top.bid_qty > 0:
        out["top_pressure"] = top.ask_qty / max(1, top.bid_qty)
    
    if ask_sum > 0:
        out["ask_share1"] = top.ask_qty / ask_sum
    
    tight = (out["spread_pct"] is not None) and (out["spread_pct"] <= config.tight_spread_pct)
    demand_dom = (out["imb_weighted"] is not None) and (out["imb_weighted"] >= config.depth_imb_min)
    
    ask_wall = False
    if out["ask_share1"] is not None and out["ask_share1"] >= config.ask_wall_dom_ratio:
        ask_wall = True
    if out["top_pressure"] is not None and out["top_pressure"] >= config.top_pressure_alert:
        ask_wall = True
    
    out["tight_spread"] = 1.0 if tight else 0.0
    out["demand_dom"] = 1.0 if demand_dom else 0.0
    out["ask_wall_strong"] = 1.0 if ask_wall else 0.0
    
    return out


# =========================
# MARKET FILTER
# =========================
def market_filter(md: MarketData, config: Config) -> bool:
    """
    فیلتر بازار: صف خرید، رنج قیمت، حجم مجاز
    """
    if not config.enable_market_filter:
        return True
    
    # Check price range (اگر allow_negative_entry فعال باشد، محدوده پایین را نادیده بگیر)
    if md.chg_pct is not None:
        if not config.allow_negative_entry:
            if md.chg_pct < config.min_price_change_pct:
                logger.info(f"Market filter: price too negative {md.chg_pct:.2f}%")
                return False
        
        if md.chg_pct > config.max_price_change_pct:
            logger.info(f"Market filter: price in queue range {md.chg_pct:.2f}%")
            return False
    
    # Check queue (صف خرید)
    if md.depth and md.depth[0].bid_qty > 0:
        queue_ratio = md.vol_today / max(1, md.depth[0].bid_qty)
        if queue_ratio < config.max_queue_ratio:
            logger.info(f"Market filter: possible buy queue {queue_ratio:.2f}")
            return False
    
    # Check min volume (شرط را شل‌تر کن: 20% به جای 30%)
    if md.avg_vol20 > 0 and md.vol_today < 0.2 * md.avg_vol20:
        logger.info("Market filter: very low volume")
        return False
    
    return True


def correlation_check(md: MarketData, config: Config) -> bool:
    """بررسی همبستگی با شاخص کل"""
    if not config.enable_index_correlation:
        return True
    
    if md.index_change_pct is None:
        return True  # اگر داده نداریم، محدودیت نگذاریم
    
    if md.index_change_pct < config.min_index_change_pct:
        logger.info(f"Correlation: index too negative {md.index_change_pct:.2f}%")
        return False
    
    return True


# =========================
# SIGNAL GENERATION
# =========================
def generate_signal(md: MarketData, config: Config) -> Signal:
    cond: Dict[str, bool] = {}
    
    # اگر allow_negative_entry فعال باشد، قیمت منفی هم مجاز است
    if config.allow_negative_entry:
        cond["price_positive"] = True  # همیشه True
    else:
        cond["price_positive"] = (md.chg_pct is not None and md.chg_pct > 0)
    
    cond["value_ge_2x_avg"] = (md.avg_val5 > 0 and md.val_today >= config.val_mult * md.avg_val5)
    cond["volume_suspicious"] = (md.avg_vol20 > 0 and md.vol_today >= config.vol_mult * md.avg_vol20)
    cond["real_power_ge_min"] = (md.real_power >= config.min_real_power)
    cond["net_real_positive"] = (md.net_real_vol > 0)
    
    # استثنا: اگر RealPower خیلی بالا باشد (>4)، شرایط را کمی شل‌تر کن
    strong_real_power = md.real_power >= 4.0
    if strong_real_power:
        # در صورت RealPower بالا، حداقل امتیاز را کاهش بده
        effective_min_score = max(2, config.min_score - 1)
    else:
        effective_min_score = config.min_score
    
    score = sum(1 for k in ["price_positive", "value_ge_2x_avg", "volume_suspicious", 
                            "real_power_ge_min", "net_real_positive"] if cond.get(k))
    
    tight_spread = bool(md.depth_metrics.get("tight_spread"))
    demand_dom = bool(md.depth_metrics.get("demand_dom"))
    ask_wall = bool(md.depth_metrics.get("ask_wall_strong"))
    
    # شرط depth: اگر RealPower خیلی قوی است، AskWall را نادیده بگیر
    if strong_real_power:
        depth_ok = (demand_dom or tight_spread)  # AskWall را ignore کن
    else:
        depth_ok = (demand_dom or tight_spread) and (not ask_wall)
    
    market_filter_pass = market_filter(md, config)
    correlation_ok = correlation_check(md, config)
    
    enter = (
        score >= effective_min_score
        and cond["real_power_ge_min"]
        and depth_ok
        and market_filter_pass
        and correlation_ok
    )
    
    return Signal(
        enter=enter,
        exit=False,
        exit_reason=None,
        score=score,
        conditions=cond,
        depth_ok=depth_ok,
        market_filter_pass=market_filter_pass,
        correlation_ok=correlation_ok
    )


# =========================
# POSITION MANAGEMENT
# =========================
def calc_qty(
    capital_rial: int,
    price: int,
    stop_loss_pct: float,
    risk_pct: float,
    max_exposure_pct: float,
) -> Tuple[int, int, int]:
    if capital_rial <= 0 or price <= 0:
        return 0, 0, 0
    
    risk_amount = capital_rial * (risk_pct / 100.0)
    per_share_risk = price * (stop_loss_pct / 100.0)
    qty_by_risk = int(risk_amount / per_share_risk) if per_share_risk > 0 else 0
    qty_by_risk = max(0, qty_by_risk)
    
    cap_amount = capital_rial * (max_exposure_pct / 100.0)
    qty_by_capital = int(cap_amount / price)
    qty_by_capital = max(0, qty_by_capital)
    
    qty_final = min(qty_by_risk, qty_by_capital)
    return qty_final, qty_by_risk, qty_by_capital


def calc_stops_with_atr(
    entry_price: int,
    atr: Optional[float],
    config: Config
) -> Tuple[int, int]:
    """Calculate stop loss and take profit, optionally using ATR"""
    if config.use_atr_stops and atr is not None and atr > 0:
        stop_distance = int(atr * config.atr_stop_multiplier)
        stop_loss = max(1, entry_price - stop_distance)
        take_profit = entry_price + int(stop_distance * 1.5)  # 1.5x reward/risk
    else:
        stop_loss = int(entry_price * (1 - config.stop_loss_pct / 100.0))
        take_profit = int(entry_price * (1 + config.take_profit_pct / 100.0))
    
    return stop_loss, take_profit


def exit_rules(
    pos: Position,
    md: MarketData,
    config: Config
) -> Tuple[bool, Optional[str]]:
    last_price = md.last_price
    
    # Update peak
    pos.peak_price = max(pos.peak_price, last_price)
    
    # Stop loss
    if last_price <= pos.stop_loss:
        return True, "STOP_LOSS"
    
    # Take profit with trailing
    if pos.peak_price >= pos.take_profit:
        trail = int(pos.peak_price * (1 - config.trail_after_profit_pct / 100.0))
        if last_price <= trail:
            return True, "TRAIL_STOP"
    
    # Net real negative
    if md.net_real_vol < 0:
        return True, "NET_REAL_NEGATIVE"
    
    # Ask wall + RP drop
    ask_wall = bool(md.depth_metrics.get("ask_wall_strong"))
    if ask_wall and md.real_power < config.rp_exit_level:
        return True, "ASK_WALL_AND_RP_DROP"
    
    return False, None


# =========================
# STATE MANAGEMENT
# =========================
def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error(f"load_state error: {e}")
            return {}
    return {}


def save_state(st: dict) -> None:
    try:
        STATE_FILE.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.error(f"save_state error: {e}")


# =========================
# METRICS & PERFORMANCE
# =========================
class PerformanceTracker:
    def __init__(self):
        self.trades: List[Dict[str, Any]] = []
        self.load()
    
    def load(self):
        if METRICS_FILE.exists():
            try:
                data = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
                self.trades = data.get("trades", [])
            except Exception:
                pass
    
    def save(self):
        try:
            data = {"trades": self.trades}
            METRICS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.error(f"save metrics error: {e}")
    
    def add_trade(self, entry_price: int, exit_price: int, qty: int, reason: str):
        pnl = (exit_price - entry_price) * qty
        pnl_pct = ((exit_price - entry_price) / entry_price) * 100.0
        
        self.trades.append({
            "entry_price": entry_price,
            "exit_price": exit_price,
            "qty": qty,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "reason": reason,
            "timestamp": now_str()
        })
        self.save()
        
        logger.info(f"Trade closed: PnL={pnl:,} ({pnl_pct:.2f}%) | Reason={reason}")
    
    def get_stats(self) -> Dict[str, Any]:
        if not self.trades:
            return {"total_trades": 0}
        
        total = len(self.trades)
        wins = sum(1 for t in self.trades if t["pnl"] > 0)
        losses = total - wins
        win_rate = (wins / total) * 100.0 if total > 0 else 0.0
        
        total_pnl = sum(t["pnl"] for t in self.trades)
        avg_pnl = total_pnl / total if total > 0 else 0
        
        return {
            "total_trades": total,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_pnl": avg_pnl
        }


# =========================
# MAIN LOGIC
# =========================
def fetch_market_data(ins_code: str, config: Config) -> MarketData:
    # Fetch MarketWatch
    mw = http_get_with_retry(
        "/tsev2/data/MarketWatchInit.aspx",
        params={"h": "0", "r": "0"},
        min_len=200,
        timeout=config.timeout,
        max_retries=config.max_retries
    )
    prices_rows, best_rows = parse_marketwatchinit(mw)
    
    pr = get_price_row(prices_rows, ins_code)
    if not pr:
        raise RuntimeError("Price row not found in MarketWatchInit")
    
    last_price = safe_int(pr[7])
    close_price = safe_int(pr[6])
    yest_price = safe_int(pr[13])
    vol_today = safe_int(pr[9])
    val_today = safe_int(pr[10])
    low_today = safe_int(pr[11])
    high_today = safe_int(pr[12])
    
    chg_pct = ((last_price - yest_price) / yest_price * 100.0) if yest_price > 0 else None
    
    # Fetch financial data
    avg_vol20, avg_val5, atr = fetch_financial_avgs(ins_code, config)
    
    # Fetch clienttype
    nums = fetch_clienttype_numbers(ins_code, config)
    real_power, net_real_vol = infer_realpower_and_netreal(nums)
    
    # Fetch depth
    depth = extract_depth(best_rows, ins_code, levels=5)
    depth_metrics = depth_metrics_advanced(depth, last_price, config)
    
    # Fetch index
    index_change_pct = get_index_change(config)
    
    return MarketData(
        ins_code=ins_code,
        last_price=last_price,
        close_price=close_price,
        yest_price=yest_price,
        vol_today=vol_today,
        val_today=val_today,
        low_today=low_today,
        high_today=high_today,
        chg_pct=chg_pct,
        avg_vol20=avg_vol20,
        avg_val5=avg_val5,
        real_power=real_power,
        net_real_vol=net_real_vol,
        depth=depth,
        depth_metrics=depth_metrics,
        index_change_pct=index_change_pct,
        atr=atr
    )


def run_once(ins_code: str, config: Config, tracker: PerformanceTracker):
    logger.info("=" * 72)
    logger.info(f"Starting cycle for {ins_code}")
    
    try:
        # Fetch all market data
        md = fetch_market_data(ins_code, config)
        
        # Generate signal
        signal = generate_signal(md, config)
        
        # Load current state
        st = load_state()
        in_pos = bool(st.get("in_position", False))
        
        # Position management
        qty = 0
        qty_by_risk = 0
        qty_by_cap = 0
        
        if md.last_price <= config.max_price_for_qty:
            qty, qty_by_risk, qty_by_cap = calc_qty(
                capital_rial=config.capital_rial,
                price=md.last_price,
                stop_loss_pct=config.stop_loss_pct,
                risk_pct=config.risk_pct,
                max_exposure_pct=config.max_exposure_pct
            )
        
        # Exit logic
        exit_now = False
        exit_reason = None
        
        if in_pos:
            pos = Position(
                entry_price=safe_int(st.get("entry_price")),
                peak_price=safe_int(st.get("peak_price")),
                qty=safe_int(st.get("qty")),
                entry_time=st.get("entry_time", ""),
                stop_loss=safe_int(st.get("stop_loss")),
                take_profit=safe_int(st.get("take_profit"))
            )
            exit_now, exit_reason = exit_rules(pos, md, config)
            
            # Update peak in state
            st["peak_price"] = pos.peak_price
            save_state(st)
        
        # State transitions
        if (not in_pos) and signal.enter:
            stop_loss, take_profit = calc_stops_with_atr(md.last_price, md.atr, config)
            
            st = {
                "in_position": True,
                "entry_price": md.last_price,
                "peak_price": md.last_price,
                "qty": qty,
                "entry_time": now_str(),
                "stop_loss": stop_loss,
                "take_profit": take_profit
            }
            save_state(st)
            in_pos = True
            logger.info(f"🟢 ENTRY: Price={md.last_price:,} | Qty={qty:,} | SL={stop_loss:,} | TP={take_profit:,}")
        
        elif in_pos and exit_now:
            entry_price = safe_int(st.get("entry_price"))
            exit_qty = safe_int(st.get("qty"))
            
            tracker.add_trade(entry_price, md.last_price, exit_qty, exit_reason)
            
            st["in_position"] = False
            st["exit_price"] = md.last_price
            st["exit_time"] = now_str()
            st["exit_reason"] = exit_reason
            save_state(st)
            in_pos = False
            logger.info(f"🔴 EXIT: Price={md.last_price:,} | Reason={exit_reason}")
        
        # Print report
        print_report(md, signal, st, in_pos, qty, qty_by_risk, qty_by_cap, config, tracker)
        
    except Exception as e:
        logger.error(f"run_once error: {e}", exc_info=True)
        raise


def print_report(
    md: MarketData,
    signal: Signal,
    st: dict,
    in_pos: bool,
    qty: int,
    qty_by_risk: int,
    qty_by_cap: int,
    config: Config,
    tracker: PerformanceTracker
):
    print("\n" + "=" * 72)
    print(f"📊 AYAR ENHANCED REPORT | {now_str()}")
    print("=" * 72)
    
    # Price info
    print(f"\n💰 PRICE DATA:")
    print(f"  Last: {md.last_price:,} | Close: {md.close_price:,} | Yest: {md.yest_price:,}")
    print(f"  Change: {fmt(md.chg_pct)}% | High: {md.high_today:,} | Low: {md.low_today:,}")
    print(f"  Index: {fmt(md.index_change_pct)}%")
    
    # Volume & Value
    print(f"\n📈 VOLUME & VALUE:")
    print(f"  Vol Today: {md.vol_today:,} | Avg20: {md.avg_vol20:,}")
    print(f"  Val Today: {md.val_today:,} | Avg5: {md.avg_val5:,}")
    
    # Client Type
    print(f"\n👥 CLIENT TYPE:")
    print(f"  RealPower: {fmt(md.real_power)} | NetRealVol: {md.net_real_vol:,}")
    
    # Depth
    print(f"\n📚 DEPTH ANALYSIS:")
    if md.depth_metrics["spread_pct"] is not None:
        print(f"  Spread: {md.depth_metrics['spread_pct']:.3f}%")
        print(f"  Imbalance: Simple={fmt(md.depth_metrics['imb_simple'])} | Weighted={fmt(md.depth_metrics['imb_weighted'])}")
        print(f"  TopPressure: {fmt(md.depth_metrics['top_pressure'])} | AskShare1: {fmt(md.depth_metrics['ask_share1'])}")
        print(f"  Flags: TightSpread={bool(md.depth_metrics['tight_spread'])} | DemandDom={bool(md.depth_metrics['demand_dom'])} | AskWall={bool(md.depth_metrics['ask_wall_strong'])}")
    else:
        print("  Depth: NA")
    
    if md.depth:
        top = md.depth[0]
        print(f"  Top Book: Bid {top.bid_price:,}({top.bid_qty:,}) | Ask {top.ask_price:,}({top.ask_qty:,})")
    
    # ATR
    if md.atr is not None:
        print(f"\n📉 ATR: {fmt(md.atr)}")
    
    # Signal conditions
    print(f"\n🎯 ENTRY CONDITIONS (Score: {signal.score}/5):")
    for k, v in signal.conditions.items():
        print(f"  {k:25s}: {'✅' if v else '❌'}")
    
    print(f"\n🚦 FILTERS:")
    print(f"  Depth OK: {'✅' if signal.depth_ok else '❌'}")
    print(f"  Market Filter: {'✅' if signal.market_filter_pass else '❌'}")
    print(f"  Correlation: {'✅' if signal.correlation_ok else '❌'}")
    
    # Position
    print(f"\n💼 POSITION:")
    pos_status = "🟢 IN POSITION" if in_pos else "⚪ OUT"
    print(f"  Status: {pos_status}")
    
    if in_pos:
        entry = safe_int(st.get("entry_price"))
        peak = safe_int(st.get("peak_price"))
        pos_qty = safe_int(st.get("qty"))
        sl = safe_int(st.get("stop_loss"))
        tp = safe_int(st.get("take_profit"))
        
        unrealized_pnl = (md.last_price - entry) * pos_qty if entry > 0 else 0
        unrealized_pct = ((md.last_price - entry) / entry * 100.0) if entry > 0 else 0.0
        
        print(f"  Entry: {entry:,} | Peak: {peak:,} | Qty: {pos_qty:,}")
        print(f"  SL: {sl:,} | TP: {tp:,}")
        print(f"  Unrealized PnL: {unrealized_pnl:,} ({unrealized_pct:.2f}%)")
    else:
        print(f"  Entry: NA")
    
    # Signal
    print(f"\n🚨 SIGNAL:")
    if signal.enter:
        print(f"  ✅ ENTER - ورود مجاز")
        if qty > 0:
            notional = qty * md.last_price
            risk_amt = int(config.capital_rial * (config.risk_pct / 100.0))
            print(f"  Qty: {qty:,} (Risk: {qty_by_risk:,} | Cap: {qty_by_cap:,})")
            print(f"  Notional: {notional:,} ریال | Risk Amount: {risk_amt:,} ریال")
    else:
        print(f"  ❌ NO ENTER - فعلاً ورود نکن")
    
    if signal.exit:
        print(f"  🔴 EXIT - خروج | Reason: {signal.exit_reason}")
    
    # Performance stats
    stats = tracker.get_stats()
    if stats["total_trades"] > 0:
        print(f"\n📊 PERFORMANCE:")
        print(f"  Total Trades: {stats['total_trades']}")
        print(f"  Wins: {stats['wins']} | Losses: {stats['losses']}")
        print(f"  Win Rate: {stats['win_rate']:.2f}%")
        print(f"  Total PnL: {stats['total_pnl']:,} ریال")
        print(f"  Avg PnL: {stats['avg_pnl']:,.0f} ریال")
    
    print("=" * 72 + "\n")


# =========================
# CLI
# =========================
def build_argparser():
    ap = argparse.ArgumentParser(description="AYAR Enhanced Trading System")
    
    # Basic
    ap.add_argument("--inscode", default="34144395039913458", help="InsCode to trade")
    ap.add_argument("--watch", action="store_true", help="Watch mode (continuous)")
    ap.add_argument("--interval", type=int, default=30, help="Watch interval in seconds")
    
    # Capital & Risk
    ap.add_argument("--capital_rial", type=int, default=500_000_000)
    ap.add_argument("--risk_pct", type=float, default=1.0)
    ap.add_argument("--max_exposure_pct", type=float, default=95.0)
    
    # Entry
    ap.add_argument("--min_real_power", type=float, default=1.5)
    ap.add_argument("--vol_mult", type=float, default=2.0)
    ap.add_argument("--val_mult", type=float, default=2.0)
    ap.add_argument("--min_score", type=int, default=3)
    ap.add_argument("--allow_negative_entry", action="store_true", help="Allow entry on negative price (برای کف‌شکن‌ها)")
    
    # Exit
    ap.add_argument("--stop_loss_pct", type=float, default=1.20)
    ap.add_argument("--take_profit_pct", type=float, default=1.80)
    ap.add_argument("--use_atr_stops", action="store_true", help="Use ATR-based stops")
    
    # Filters
    ap.add_argument("--enable_market_filter", action="store_true", default=True)
    ap.add_argument("--enable_index_correlation", action="store_true", default=True)
    
    # System
    ap.add_argument("--timeout", type=int, default=20)
    ap.add_argument("--max_retries", type=int, default=3)
    
    return ap


def main():
    args = build_argparser().parse_args()
    
    # Build config from args
    config = Config(
        capital_rial=args.capital_rial,
        risk_pct=args.risk_pct,
        max_exposure_pct=args.max_exposure_pct,
        min_real_power=args.min_real_power,
        vol_mult=args.vol_mult,
        val_mult=args.val_mult,
        min_score=args.min_score,
        allow_negative_entry=args.allow_negative_entry,
        stop_loss_pct=args.stop_loss_pct,
        take_profit_pct=args.take_profit_pct,
        use_atr_stops=args.use_atr_stops,
        enable_market_filter=args.enable_market_filter,
        enable_index_correlation=args.enable_index_correlation,
        timeout=args.timeout,
        max_retries=args.max_retries
    )
    
    tracker = PerformanceTracker()
    
    logger.info("=" * 72)
    logger.info("AYAR ENHANCED TRADING SYSTEM STARTED")
    logger.info(f"Config: {asdict(config)}")
    logger.info("=" * 72)
    
    while True:
        try:
            run_once(args.inscode, config, tracker)
        except Exception as e:
            logger.error(f"Main loop error: {e}", exc_info=True)
        
        if not args.watch:
            break
        
        time.sleep(max(5, args.interval))


if __name__ == "__main__":
    main()