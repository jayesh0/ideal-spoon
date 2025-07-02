import logging
import os
import sys
import time
from datetime import datetime, timezone, date, timedelta

# Rich library for CLI UI
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text
from rich.panel import Panel
from rich.layout import Group

# KiteConnect API
try:
    from kiteconnect import KiteConnect
except ImportError:
    print("KiteConnect library not found. Please install it: pip install kiteconnect")
    sys.exit(1)

# --- Configuration ---

# --- API Credentials ---
# Try to get from environment variables first
KITE_API_KEY_ENV_NAME = "KITE_API_KEY"
KITE_API_SECRET_ENV_NAME = "KITE_API_SECRET"

API_KEY_DEFAULT = "YOUR_API_KEY"        # Default placeholder if env var not found
API_SECRET_DEFAULT = "YOUR_API_SECRET"  # Default placeholder if env var not found

# --- Trading Parameters ---
# -- NIFTY Parameters (Example) --
UNDERLYING_SYMBOL = "NIFTY 50"       # Underlying instrument (e.g., "NIFTY 50", "NIFTY BANK", "SENSEX")
STRIKE_DIFFERENCE = 50               # Difference between consecutive strikes (50 for NIFTY, 100 for BANKNIFTY/SENSEX)
OPTIONS_COUNT = 2                    # Number of ITM/OTM strikes to fetch on each side of ATM (ATM-2, ATM-1, ATM, ATM+1, ATM+2 = 5 levels total)

# --- Exchange Configuration ---
# Use KiteConnect attributes for exchange names
EXCHANGE_NFO_OPTIONS = KiteConnect.EXCHANGE_NFO  # Exchange for NFO options contracts (e.g., NFO for NIFTY/BANKNIFTY, BFO for SENSEX)
EXCHANGE_LTP = KiteConnect.EXCHANGE_NSE      # Exchange for fetching LTP of the underlying (e.g., NSE for NIFTY 50, BSE for SENSEX)

# To use SENSEX instead, comment NIFTY params and uncomment below (and adjust KiteConnect exchanges if needed)
# -- SENSEX Parameters --
# UNDERLYING_SYMBOL = "SENSEX"
# STRIKE_DIFFERENCE = 100
# OPTIONS_COUNT = 2
# EXCHANGE_NFO_OPTIONS = KiteConnect.EXCHANGE_BFO # BSE Futures and Options
# EXCHANGE_LTP = KiteConnect.EXCHANGE_BSE


# --- Data Fetching Parameters ---
HISTORICAL_DATA_MINUTES = 40         # How many minutes of historical data to fetch for OI calculation (should be > max interval)
OI_CHANGE_INTERVALS_MIN = (10, 15, 30) # Past intervals (in minutes) to calculate OI change from latest OI

# --- Display and Logging ---
REFRESH_INTERVAL_SECONDS = 60        # How often to refresh the data and tables (in seconds)
LOG_FILE_NAME = "oi_tracker.log"     # Name of the log file
# Logging level for the log file (DEBUG, INFO, WARNING, ERROR, CRITICAL)
# Ensure this is a string that getattr can use, e.g., "INFO"
FILE_LOG_LEVEL_STR = "INFO"
PCT_CHANGE_THRESHOLDS = {            # Thresholds for highlighting OI % change (interval_in_min: percentage)
    10: 10.0,
    15: 15.0,
    30: 25.0
}
# Alert sound configuration
ALERT_SOUND_ENABLED = True # Set to False to disable sound
ALERT_SOUND_FILE_PATH = "/Users/vibhu/zd/siren-alert-96052.mp3" # macOS specific example
# For cross-platform sound, consider libraries like 'playsound' (pip install playsound)
# and then use: from playsound import playsound; playsound(ALERT_SOUND_FILE_PATH)

# ==============================================================================
# --- END OF CONFIGURATION ---
# ==============================================================================

# --- Global Initializations ---
# Setup file logging
try:
    log_level = getattr(logging, FILE_LOG_LEVEL_STR.upper(), logging.INFO)
except AttributeError:
    print(f"Invalid FILE_LOG_LEVEL_STR: {FILE_LOG_LEVEL_STR}. Defaulting to INFO.")
    log_level = logging.INFO

logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(levelname)s - %(module)s.%(funcName)s - %(message)s',
    filename=LOG_FILE_NAME,
    filemode='a'  # Append to the log file
)

# Derive underlying prefix (e.g., NIFTY from "NIFTY 50") for instrument searching
# This assumes the symbol format is "PREFIX ..." like "NIFTY 50" or "BANKNIFTY INDEX"
UNDERLYING_PREFIX = UNDERLYING_SYMBOL.split(" ")[0].upper()

# Retrieve API keys from environment variables or use default placeholders
api_key_to_use = os.getenv(KITE_API_KEY_ENV_NAME, API_KEY_DEFAULT)
api_secret_to_use = os.getenv(KITE_API_SECRET_ENV_NAME, API_SECRET_DEFAULT)


# Global KiteConnect and Rich Console instances
# Initialize KiteConnect with reduced library logging to prevent console clutter from the library itself
kite = KiteConnect(api_key=api_key_to_use, logging_level=logging.WARNING) # Reduce Kite lib's own logging
console = Console() # For Rich text and table display


# --- Core Functions ---

def get_atm_strike(kite_obj: KiteConnect, underlying_sym: str, exch_for_ltp: str, strike_diff: int):
    """
    Fetches the Last Traded Price (LTP) for the underlying symbol and calculates the At-The-Money (ATM) strike.
    """
    try:
        ltp_instrument = f"{exch_for_ltp}:{underlying_sym}"
        ltp_data = kite_obj.ltp(ltp_instrument)

        if not ltp_data or ltp_instrument not in ltp_data or 'last_price' not in ltp_data[ltp_instrument]:
            logging.error(f"LTP data not found or incomplete for {ltp_instrument}. Response: {ltp_data}")
            return None

        ltp = ltp_data[ltp_instrument]['last_price']
        atm_strike = round(ltp / strike_diff) * strike_diff
        logging.debug(f"LTP for {underlying_sym}: {ltp}, Calculated ATM strike: {atm_strike}")
        return atm_strike
    except Exception as e:
        logging.error(f"Error in get_atm_strike for {underlying_sym}: {e}", exc_info=True)
        return None

def get_nearest_weekly_expiry(instruments: list, underlying_prefix_str: str, exchange_nfo: str):
    """
    Finds the nearest future weekly expiry date and symbol prefix for the given underlying.
    """
    today = date.today()
    possible_expiries = {} # Store expiry_date: trading_symbol_prefix_part

    logging.info(f"Searching for nearest weekly expiry for {underlying_prefix_str} on {exchange_nfo} among {len(instruments)} instruments.")

    for inst in instruments:
        if inst['name'] == underlying_prefix_str and inst['exchange'] == exchange_nfo:
            inst_expiry_date = inst['expiry']
            # Ensure expiry is a date object and is in the future or today
            if isinstance(inst_expiry_date, date) and inst_expiry_date >= today:
                if inst_expiry_date not in possible_expiries:
                    # Derive the common prefix part of the trading symbol for that expiry
                    # Example: NIFTY23OCT for NIFTY23OCT19500CE
                    # This logic might need adjustment if symbol formats vary significantly
                    # For NIFTY/BANKNIFTY, it's usually UNDERLYINGYYMON (e.g., NIFTY23OCT)
                    # For weekly, it's often UNDERLYINGYYMDD (e.g., NIFTY23O05) or similar
                    # Let's assume a common pattern: PREFIX + YY + single char for month/week + day/week indicator
                    # A more robust way is to find options for a strike and get their common prefix.
                    # For now, a simplified approach:
                    # NIFTY23OCT19500CE -> NIFTY23OCT
                    # BANKNIFTY2390745000CE -> BANKNIFTY23907 (example for weekly)
                    # The provided script used a fixed length slice which might be fragile.
                    # A safer way: find first strike, take its symbol, remove strike and CE/PE.
                    # Let's try to match based on year and month primarily for weekly.
                    # Example: NIFTY23OCT, NIFTY23NOV, NIFTY23DEC for monthly
                    # NIFTY23O05, NIFTY23O12 for weekly (NIFTY YY M DD where M is A-L for Jan-Dec, or O for Oct weekly)
                    # This part is tricky without knowing the exact symbol convention for all weeklies.
                    # The provided script's logic was: trading_symbol_of_nearest_expiry[0:len(underlying_prefix_str)+5]
                    # e.g., for NIFTY23OCT19500CE, it gives NIFTY23OCT. This seems okay for monthly.
                    # For weekly like NIFTY23O1219500CE, it would give NIFTY23O12.

                    # We need a way to get the part like "NIFTY23OCT" or "NIFTY23O12"
                    # Let's find the part of the tradingsymbol before the strike
                    symbol = inst['tradingsymbol']
                    strike_str = str(int(inst['strike']))

                    # Find strike in symbol, then take prefix
                    strike_pos = symbol.find(strike_str)
                    if strike_pos > 0:
                        symbol_prefix_part = symbol[:strike_pos]
                        possible_expiries[inst_expiry_date] = symbol_prefix_part
                    else:
                        # Fallback or more complex logic might be needed if strike isn't directly in symbol string like that
                        # For now, log a warning if this happens
                        logging.warning(f"Could not easily determine symbol prefix for {symbol} with strike {strike_str}")

    if not possible_expiries:
        logging.error(f"No future expiries found for {underlying_prefix_str} on {exchange_nfo}.")
        return None

    nearest_expiry_date = sorted(list(possible_expiries.keys()))[0]
    symbol_prefix = possible_expiries[nearest_expiry_date]

    logging.info(f"Nearest weekly expiry for {underlying_prefix_str}: {nearest_expiry_date}, Symbol Prefix Part: {symbol_prefix}")
    return {"expiry": nearest_expiry_date, "symbol_prefix": symbol_prefix}


def get_relevant_option_details(instruments: list, atm_strike_val: float, expiry_dt: date,
                                strike_diff_val: int, opt_count: int, underlying_prefix_str: str,
                                symbol_prefix_for_expiry: str, exchange_nfo: str):
    """
    Identifies relevant ITM, ATM, and OTM Call/Put option contract details.
    """
    relevant_options = {}
    if not expiry_dt or atm_strike_val is None:
        logging.error("Expiry date or ATM strike is None, cannot fetch option details.")
        return relevant_options

    logging.debug(f"Searching for options with expiry: {expiry_dt}, ATM strike: {atm_strike_val}, Symbol Prefix: {symbol_prefix_for_expiry}")

    for i in range(-opt_count, opt_count + 1): # e.g., -2, -1, 0, 1, 2 for opt_count = 2
        current_strike = atm_strike_val + (i * strike_diff_val)

        # Construct the expected trading symbol prefix for this strike
        # e.g., symbol_prefix_for_expiry (like NIFTY23OCT) + strike (19500)
        ce_symbol_pattern_base = f"{symbol_prefix_for_expiry}{int(current_strike)}CE"
        pe_symbol_pattern_base = f"{symbol_prefix_for_expiry}{int(current_strike)}PE"

        found_ce, found_pe = None, None
        for inst in instruments:
            if inst['name'] == underlying_prefix_str and \
               inst['strike'] == current_strike and \
               inst['expiry'] == expiry_dt and \
               inst['exchange'] == exchange_nfo:

                # Check if the instrument's trading symbol matches the expected pattern
                if inst['instrument_type'] == 'CE' and inst['tradingsymbol'].startswith(symbol_prefix_for_expiry) and inst['tradingsymbol'].endswith('CE'):
                    if inst['tradingsymbol'] == ce_symbol_pattern_base : # Exact match preferred
                         found_ce = inst
                    elif not found_ce : # Fallback if exact match is not found first (e.g. slight variations)
                         found_ce = inst # Take first match for this strike/expiry/type
                elif inst['instrument_type'] == 'PE' and inst['tradingsymbol'].startswith(symbol_prefix_for_expiry) and inst['tradingsymbol'].endswith('PE'):
                    if inst['tradingsymbol'] == pe_symbol_pattern_base:
                        found_pe = inst
                    elif not found_pe:
                        found_pe = inst

            if found_ce and found_pe: # Optimization
                break

        key_suffix = _get_key_suffix(i, opt_count) # atm, itm1, otm1 etc.

        if found_ce:
            relevant_options[f"{key_suffix}_ce"] = {
                'tradingsymbol': found_ce['tradingsymbol'],
                'instrument_token': found_ce['instrument_token'],
                'strike': current_strike
            }
        else:
            logging.warning(f"CE option not found for strike {current_strike}, expiry {expiry_dt} with pattern {ce_symbol_pattern_base}")

        if found_pe:
            relevant_options[f"{key_suffix}_pe"] = {
                'tradingsymbol': found_pe['tradingsymbol'],
                'instrument_token': found_pe['instrument_token'],
                'strike': current_strike
            }
        else:
            logging.warning(f"PE option not found for strike {current_strike}, expiry {expiry_dt} with pattern {pe_symbol_pattern_base}")

    logging.debug(f"Relevant option details identified: {len(relevant_options)} contracts.")
    return relevant_options

def fetch_historical_oi_data(kite_obj: KiteConnect, option_details_dict: dict,
                             minutes_of_data: int = HISTORICAL_DATA_MINUTES):
    """
    Fetches historical OI data (minute interval) for the provided option contracts.
    """
    historical_oi_store = {}
    if not option_details_dict:
        logging.warning("No option details provided to fetch_historical_oi_data.")
        return historical_oi_store

    to_date = datetime.now()
    from_date = to_date - timedelta(minutes=minutes_of_data)
    # Kite API historical data is EOD for free plans. Live data might need different handling or paid API.
    # Assuming this is for EOD or where minute data is available.
    # For live OI, KiteTicker might be needed if historical_data doesn't update intra-minute frequently.
    # The problem states "fetch the OI using historical_data api", so we stick to that.
    from_date_str = from_date.strftime("%Y-%m-%d %H:%M:%S") # Using HH:MM:SS
    to_date_str = to_date.strftime("%Y-%m-%d %H:%M:%S")


    logging.debug(f"Fetching historical data from {from_date_str} to {to_date_str}")

    for option_key, details in option_details_dict.items():
        instrument_token = details.get('instrument_token')
        tradingsymbol = details.get('tradingsymbol')

        if not instrument_token:
            logging.warning(f"Missing instrument_token for {option_key} ({tradingsymbol}). Skipping.")
            historical_oi_store[option_key] = []
            continue

        try:
            logging.debug(f"Fetching historical OI for {tradingsymbol} (Token: {instrument_token})")
            # Ensure dates are strings. Max 100 days for minute data.
            data = kite_obj.historical_data(instrument_token, from_date_str, to_date_str, interval="minute", oi=True)
            historical_oi_store[option_key] = data
            logging.debug(f"Fetched {len(data)} records for {tradingsymbol}")
        except Exception as e:
            logging.error(f"Error fetching historical OI for {tradingsymbol} (Token: {instrument_token}): {e}", exc_info=True)
            historical_oi_store[option_key] = []
    return historical_oi_store

def find_oi_at_timestamp(historical_candles: list, target_time: datetime,
                          latest_oi_and_time: tuple = None):
    """
    Finds Open Interest (OI) at or just before a specific target_time from historical candles.
    Candles are assumed to be sorted oldest to newest.
    target_time should be timezone-aware if candles['date'] is timezone-aware.
    Kite historical data usually returns timezone-aware UTC datetimes.
    """
    if not historical_candles:
        return None

    # Ensure target_time is timezone-aware, matching candle['date'] (usually UTC from Kite)
    if target_time.tzinfo is None: # If target_time is naive
        # Assuming historical_candles[0]['date'] is timezone-aware (e.g. UTC)
        # Make target_time aware of the same timezone for comparison
        if historical_candles and historical_candles[0]['date'].tzinfo:
             target_time = target_time.replace(tzinfo=historical_candles[0]['date'].tzinfo)
        else: # Fallback: if candle date is also naive, or no candles, assume UTC for safety
             target_time = target_time.replace(tzinfo=timezone.utc)


    selected_oi = None
    # Iterate backwards
    for candle in reversed(historical_candles):
        candle_time = candle['date'] # This is a datetime object from Kite, usually UTC

        # Ensure candle_time is also timezone-aware for comparison
        if candle_time.tzinfo is None : # Should not happen with Kite data typically
            candle_time = candle_time.replace(tzinfo=target_time.tzinfo)


        if candle_time <= target_time:
            # If latest_oi_and_time is provided, ensure we don't pick a candle
            # whose timestamp is later than the latest_oi_timestamp from the most current data point.
            if latest_oi_and_time and latest_oi_and_time[1] and candle_time > latest_oi_and_time[1]:
                continue
            selected_oi = candle.get('oi')
            break # Found the most recent candle at or before target_time

    return selected_oi


def calculate_oi_differences(raw_historical_data_store: dict, intervals_min: tuple):
    """
    Calculates OI differences between the latest OI and OI at specified past intervals.
    """
    oi_differences_report = {}
    # Use a consistent, timezone-aware current time (UTC) for all calculations in this batch
    current_processing_time = datetime.now(timezone.utc)
    logging.debug(f"Calculating OI differences based on current time: {current_processing_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    for option_key, candles_list in raw_historical_data_store.items():
        oi_differences_report[option_key] = {}

        latest_oi, latest_oi_timestamp = None, None
        if candles_list:
            latest_candle = candles_list[-1] # Assumes candles sorted: oldest to newest
            latest_oi = latest_candle.get('oi')
            latest_oi_timestamp = latest_candle.get('date') # datetime object, typically UTC

        oi_differences_report[option_key]['latest_oi'] = latest_oi
        oi_differences_report[option_key]['latest_oi_timestamp'] = latest_oi_timestamp

        if latest_oi is None:
            for interval in intervals_min:
                oi_differences_report[option_key][f'pct_diff_{interval}m'] = None
            continue

        for interval in intervals_min:
            target_past_time = current_processing_time - timedelta(minutes=interval)

            past_oi = find_oi_at_timestamp(
                candles_list,
                target_past_time,
                latest_oi_and_time=(latest_oi, latest_oi_timestamp)
            )

            pct_oi_change = None
            if past_oi is not None and latest_oi is not None:
                if past_oi != 0:
                    abs_oi_diff = latest_oi - past_oi
                    pct_oi_change = (abs_oi_diff / past_oi) * 100
                # else: pct_oi_change remains None if past_oi is 0 (infinite change or undefined)

            oi_differences_report[option_key][f'pct_diff_{interval}m'] = pct_oi_change

    logging.debug("OI differences calculation complete.")
    return oi_differences_report

def _get_key_suffix(index_from_atm: int, total_options_one_side: int) -> str:
    """
    Helper to determine option key suffix (atm, itmX, otmX).
    """
    if index_from_atm == 0:
        return "atm"
    elif index_from_atm < 0:
        return f"itm{-index_from_atm}"
    else:
        return f"otm{index_from_atm}"

def play_alert_sound():
    """Plays an alert sound if enabled and file exists."""
    if ALERT_SOUND_ENABLED:
        if os.path.exists(ALERT_SOUND_FILE_PATH):
            try:
                # This is macOS specific. For cross-platform, use a library like playsound.
                # from playsound import playsound
                # playsound(ALERT_SOUND_FILE_PATH)
                if sys.platform == "darwin": # macOS
                    os.system(f"afplay '{ALERT_SOUND_FILE_PATH}'")
                    logging.info(f"Played alert sound: {ALERT_SOUND_FILE_PATH}")
                elif sys.platform == "win32": # Windows
                    # Example: using winsound (built-in)
                    # import winsound
                    # winsound.PlaySound(ALERT_SOUND_FILE_PATH, winsound.SND_FILENAME)
                    logging.info(f"Alert sound triggered for Windows (winsound placeholder). Path: {ALERT_SOUND_FILE_PATH}")
                elif sys.platform.startswith("linux"): # Linux
                    # Example: using os.system with a common player like aplay or paplay
                    # os.system(f"aplay '{ALERT_SOUND_FILE_PATH}'")
                    logging.info(f"Alert sound triggered for Linux (aplay placeholder). Path: {ALERT_SOUND_FILE_PATH}")
                else:
                    logging.warning(f"Alert sound configured, but no player implemented for platform: {sys.platform}")
            except Exception as e:
                logging.error(f"Failed to play alert sound {ALERT_SOUND_FILE_PATH}: {e}", exc_info=True)
        else:
            logging.warning(f"Alert sound file not found: {ALERT_SOUND_FILE_PATH}")

def generate_options_tables(oi_report: dict, contract_details: dict, current_atm_strike: float,
                            strike_step: int, num_strikes_each_side: int,
                            change_intervals_list: tuple):
    """
    Generates Rich Tables for Call and Put options OI analysis.
    """
    if current_atm_strike is None:
        logging.error("Cannot generate tables: current_atm_strike is None.")
        return Panel("[bold red]ATM Strike could not be determined. Tables cannot be generated.[/bold red]", title="Error", border_style="red")

    time_now_str = datetime.now().strftime('%H:%M:%S')

    call_table_title = f"CALL Options OI ({UNDERLYING_SYMBOL} - ATM: {int(current_atm_strike)}) @ {time_now_str}"
    call_table = Table(title=call_table_title, show_lines=True, expand=True)

    put_table_title = f"PUT Options OI ({UNDERLYING_SYMBOL} - ATM: {int(current_atm_strike)}) @ {time_now_str}"
    put_table = Table(title=put_table_title, show_lines=True, expand=True)

    cols = ["Strike", "Symbol", "Latest OI", "OI Time"]
    for interval in change_intervals_list:
        cols.append(f"OI %Chg ({interval}m)")

    for col_name in cols:
        call_table.add_column(col_name, justify="right")
        put_table.add_column(col_name, justify="right")

    total_call_data_cells = 0
    total_call_threshold_breached = 0
    total_put_data_cells = 0
    total_put_threshold_breached = 0

    # num_strikes_each_side = 2 means -2, -1, 0, 1, 2 (5 rows)
    for i in range(-num_strikes_each_side, num_strikes_each_side + 1):
        strike_val = current_atm_strike + (i * strike_step)
        key_suffix = _get_key_suffix(i, num_strikes_each_side)

        # --- Call Option Row ---
        option_key_ce = f"{key_suffix}_ce"
        ce_data = oi_report.get(option_key_ce, {})
        ce_contract = contract_details.get(option_key_ce, {})
        ce_strike_display = str(int(ce_contract.get('strike', strike_val)))
        ce_strike_style = "cyan" if i == 0 else ("green" if i < 0 else "red") # ITM Calls are lower strikes
        ce_latest_oi = ce_data.get('latest_oi')
        ce_latest_oi_time = ce_data.get('latest_oi_timestamp')
        ce_row_data = [
            Text(ce_strike_display, style=ce_strike_style),
            ce_contract.get('tradingsymbol', 'N/A'),
            f"{ce_latest_oi:,}" if ce_latest_oi is not None else "N/A",
            ce_latest_oi_time.strftime("%H:%M:%S %Z") if ce_latest_oi_time and isinstance(ce_latest_oi_time, datetime) else "N/A"
        ]
        for interval in change_intervals_list:
            total_call_data_cells += 1
            pct_oi_change = ce_data.get(f'pct_diff_{interval}m')
            formatted_pct_str = f"{pct_oi_change:+.2f}%" if pct_oi_change is not None else "N/A"
            cell_text = Text(formatted_pct_str)
            if pct_oi_change is not None and interval in PCT_CHANGE_THRESHOLDS:
                if abs(pct_oi_change) > PCT_CHANGE_THRESHOLDS[interval]:
                    cell_text.stylize("bold red")
                    total_call_threshold_breached += 1
            ce_row_data.append(cell_text)
        call_table.add_row(*ce_row_data)

        # --- Put Option Row ---
        option_key_pe = f"{key_suffix}_pe"
        pe_data = oi_report.get(option_key_pe, {})
        pe_contract = contract_details.get(option_key_pe, {})
        pe_strike_display = str(int(pe_contract.get('strike', strike_val)))
        pe_strike_style = "cyan" if i == 0 else ("green" if i > 0 else "red") # ITM Puts are higher strikes
        pe_latest_oi = pe_data.get('latest_oi')
        pe_latest_oi_time = pe_data.get('latest_oi_timestamp')
        pe_row_data = [
            Text(pe_strike_display, style=pe_strike_style),
            pe_contract.get('tradingsymbol', 'N/A'),
            f"{pe_latest_oi:,}" if pe_latest_oi is not None else "N/A",
            pe_latest_oi_time.strftime("%H:%M:%S %Z") if pe_latest_oi_time and isinstance(pe_latest_oi_time, datetime) else "N/A"
        ]
        for interval in change_intervals_list:
            total_put_data_cells += 1
            pct_oi_change = pe_data.get(f'pct_diff_{interval}m')
            formatted_pct_str = f"{pct_oi_change:+.2f}%" if pct_oi_change is not None else "N/A"
            cell_text = Text(formatted_pct_str)
            if pct_oi_change is not None and interval in PCT_CHANGE_THRESHOLDS:
                if abs(pct_oi_change) > PCT_CHANGE_THRESHOLDS[interval]:
                    cell_text.stylize("bold red")
                    total_put_threshold_breached += 1
            pe_row_data.append(cell_text)
        put_table.add_row(*pe_row_data)

    # Check for alert sound condition
    alert_triggered_this_cycle = False
    if total_call_data_cells > 0 and (total_call_threshold_breached / total_call_data_cells) > 0.5:
        logging.info(f"Alert condition met for CALLs: {total_call_threshold_breached}/{total_call_data_cells} cells breached threshold.")
        alert_triggered_this_cycle = True
    if total_put_data_cells > 0 and (total_put_threshold_breached / total_put_data_cells) > 0.5:
        logging.info(f"Alert condition met for PUTs: {total_put_threshold_breached}/{total_put_data_cells} cells breached threshold.")
        alert_triggered_this_cycle = True

    if alert_triggered_this_cycle:
        play_alert_sound()

    return Group(call_table, put_table)


def run_analysis_iteration(kite_conn: KiteConnect, nfo_instr_list: list, nearest_expiry_info: dict):
    """
    Performs one complete iteration of fetching data, calculating, and generating tables.
    """
    try:
        logging.debug("Starting new analysis iteration.")
        current_atm_strike = get_atm_strike(kite_conn, UNDERLYING_SYMBOL, EXCHANGE_LTP, STRIKE_DIFFERENCE)

        if not current_atm_strike:
            logging.error("Could not determine ATM strike for this iteration.")
            return Panel("[bold red]Error: Could not determine ATM strike. Check logs. Waiting for next refresh.[/bold red]", title="Update Error", border_style="red")

        if not nearest_expiry_info or 'expiry' not in nearest_expiry_info or 'symbol_prefix' not in nearest_expiry_info:
             logging.error("Nearest expiry date or symbol prefix is not available.")
             return Panel("[bold red]Error: Nearest expiry info not available. Critical error.[/bold red]", title="Update Error", border_style="red")

        nearest_exp_date = nearest_expiry_info['expiry']
        symbol_prefix_for_expiry = nearest_expiry_info['symbol_prefix']

        option_contract_details = get_relevant_option_details(
            nfo_instr_list, current_atm_strike, nearest_exp_date,
            STRIKE_DIFFERENCE, OPTIONS_COUNT, UNDERLYING_PREFIX, symbol_prefix_for_expiry, EXCHANGE_NFO_OPTIONS
        )

        if not option_contract_details:
            logging.warning(f"Could not retrieve relevant option contracts for ATM {int(current_atm_strike)} and expiry {nearest_exp_date}.")
            return Panel(f"[bold yellow]Warning: No relevant option contracts found for ATM {int(current_atm_strike)}, Expiry {nearest_exp_date}. Waiting for next refresh.[/bold yellow]", title="Update Warning", border_style="yellow")

        raw_historical_oi_data = fetch_historical_oi_data(kite_conn, option_contract_details)
        oi_change_data = calculate_oi_differences(raw_historical_oi_data, OI_CHANGE_INTERVALS_MIN)

        table_group = generate_options_tables(
            oi_change_data, option_contract_details, current_atm_strike,
            STRIKE_DIFFERENCE, OPTIONS_COUNT, OI_CHANGE_INTERVALS_MIN
        )
        logging.debug("Analysis iteration completed successfully.")
        return table_group

    except Exception as e:
        logging.error(f"Exception during analysis iteration: {e}", exc_info=True)
        return Panel(f"[bold red]An error occurred during data refresh: {str(e)}. Check logs.[/bold red]", title="Update Error", border_style="red")


def main():
    """
    Main function to run the OI Tracker script.
    """
    console.rule(f"[bold blue]OI Tracker for {UNDERLYING_SYMBOL}[/bold blue]")
    console.print(f"Logging to: [cyan]{LOG_FILE_NAME}[/cyan]")

    if api_key_to_use == API_KEY_DEFAULT or api_secret_to_use == API_SECRET_DEFAULT:
        console.print(f"[yellow]Warning: Using default placeholder API Key/Secret.[/yellow]")
        console.print(f"[yellow]Set '{KITE_API_KEY_ENV_NAME}' and '{KITE_API_SECRET_ENV_NAME}' environment variables, or update defaults in script.[/yellow]")
        logging.warning("Using default placeholder API Key/Secret.")

    if ALERT_SOUND_ENABLED and not os.path.exists(ALERT_SOUND_FILE_PATH):
        console.print(f"[yellow]Warning: Alert sound is enabled, but sound file not found at '{ALERT_SOUND_FILE_PATH}'. Alerts will be silent.[/yellow]")
        logging.warning(f"Alert sound enabled but file not found: {ALERT_SOUND_FILE_PATH}")


    try:
        access_token = os.getenv("KITE_ACCESS_TOKEN")
        if access_token:
            kite.set_access_token(access_token)
            console.print("[green]Using access token from KITE_ACCESS_TOKEN environment variable.[/green]")
        else:
            login_url = kite.login_url()
            console.print(f"Kite Login URL: [link={login_url}]{login_url}[/link]")
            request_token = console.input("[bold cyan]Enter Request Token from the login URL: [/bold cyan]").strip()
            if not request_token:
                console.print("[bold red]No request token entered. Exiting.[/bold red]")
                sys.exit(1)

            session_data = kite.generate_session(request_token, api_secret=api_secret_to_use)
            kite.set_access_token(session_data["access_token"])
            console.print("[bold green]Kite API session generated successfully![/bold green]")
            console.print(f"[dim]To skip login next time, set KITE_ACCESS_TOKEN='{session_data['access_token']}' environment variable.[/dim]")

        profile = kite.profile()
        console.print(f"[green]Connected for user: {profile.get('user_id')} ({profile.get('user_name')})[/green]")

        console.print(f"Fetching NFO instruments list for {EXCHANGE_NFO_OPTIONS} (once)...")
        nfo_instruments = kite.instruments(EXCHANGE_NFO_OPTIONS)
        if not nfo_instruments:
            console.print(f"[bold red]Failed to fetch NFO instruments from {EXCHANGE_NFO_OPTIONS}. Exiting.[/bold red]")
            sys.exit(1)
        logging.info(f"Fetched {len(nfo_instruments)} NFO instruments for {EXCHANGE_NFO_OPTIONS}.")

        nearest_expiry_details = get_nearest_weekly_expiry(nfo_instruments, UNDERLYING_PREFIX, EXCHANGE_NFO_OPTIONS)
        if not nearest_expiry_details:
            console.print(f"[bold red]Could not determine nearest weekly expiry for {UNDERLYING_PREFIX}. Exiting.[/bold red]")
            sys.exit(1)

        console.print(f"Tracking options for expiry: [bold magenta]{nearest_expiry_details['expiry'].strftime('%d-%b-%Y')}[/bold magenta], Symbol Prefix: [bold magenta]{nearest_expiry_details['symbol_prefix']}[/bold magenta]")
        console.print(f"Refresh interval: {REFRESH_INTERVAL_SECONDS}s. Options per side: {OPTIONS_COUNT}. Intervals: {OI_CHANGE_INTERVALS_MIN} mins.")
        console.rule("[bold blue]Live Updates Starting[/bold blue]")


        with Live(console=console, refresh_per_second=4, auto_refresh=False) as live:
            while True:
                logging.info("Starting new live update cycle.")
                display_content = run_analysis_iteration(kite, nfo_instruments, nearest_expiry_details)
                live.update(display_content, refresh=True)
                logging.info(f"Live display updated. Waiting for {REFRESH_INTERVAL_SECONDS} seconds.")
                time.sleep(REFRESH_INTERVAL_SECONDS)

    except KiteConnect.exceptions.TokenException as te:
        console.print(f"[bold red]Token Exception: {te}. Invalid or expired request/access token or session issues. Restart and re-login.[/bold red]")
        logging.critical(f"Token Exception: {te}", exc_info=True)
    except KiteConnect.exceptions.InputException as ie:
        console.print(f"[bold red]Input Error: {ie}. Incorrect API Key/Secret or parameters. Check config.[/bold red]")
        logging.critical(f"Input Exception: {ie}", exc_info=True)
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Script terminated by user (Ctrl+C).[/bold yellow]")
        logging.info("Script terminated by user.")
    except Exception as e:
        console.print(f"[bold red]\nAn unexpected critical error occurred: {e}[/bold red]")
        logging.critical(f"Unexpected critical error in main: {e}", exc_info=True)
    finally:
        logging.info("oi_tracker.py script execution ended.")
        console.rule("[bold blue]OI Tracker Script Finished[/bold blue]")

if __name__ == "__main__":
    main()
