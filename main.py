# btc-cent-forecast-01 entry: persistence centre, dislocation-clipped.
#
# The forecast for each case is the checkpoint's own close (persistence) --
# under absolute error the honest point forecast is the conditional median of
# the price 240 minutes ahead, and walk-forward tests on the practice cases
# showed the empirical median 240-minute change, short-horizon drift and
# mean reversion toward a 24-hour average all flip sign between evaluation
# windows: they help on the interior walk-forward samples and hurt on the
# actual checkpoint forecasts (or the reverse), so none of them is a robust
# edge. The centre therefore stays exactly at the last price.
#
# One exception is kept, and it only ever moves the guess when the last close
# is displaced far from where the price has just been trading: this price is
# a single Orca pool's quote, and the rules warn that an unusually large trade
# there moves it "for a moment". A dislocation like that reverts within the
# horizon, so a last close sitting far outside the recent level is a bad
# centre. The guess is therefore clipped to the trailing 21-minute median
# plus or minus a band B:
#
#   B = max(0.002 * last, 80 * 1.4826 * median(|1-minute changes| over the
#       last 240 minutes))
#
# The 0.2%-of-price floor keeps the band sane when the market goes flat
# (the median 1-minute change collapses to ~$0.01 then, which would make
# any fixed multiple of it fire on noise); the MAD term widens the band
# when the market is genuinely jumpy. In normal conditions the last close
# sits inside the band and the forecast is exactly persistence.
#
# Walk-forward evidence (truncate each practice case's history, forecast
# +240 min, score MAE in cents vs the recorded continuation):
#   persistence            42653.54 cents (66 samples, 240-min step)
#   this entry (clip, E)   36032.80 cents  (-15.5%, same 66 samples)
#   persistence            46528.96 cents (258 samples, 60-min step)
#   this entry (clip, E)   44891.53 cents  (-3.5%, same 258 samples)
#   persistence            43332.27 cents (63 forecasts, continuous union)
#   this entry (clip, E)   41997.67 cents  (-3.1%, same 63 forecasts)
#   on the 5 practice answers known locally the clip never fires, so the
#   entry scores exactly persistence there (47054.66 cents over those 5).
# The big walk-forward gain is mostly one real dislocation in the recording
# (price fell ~$1,264 below its 21-minute median in a minute and was back
# near it 4 hours later); the clip's value is that single free option, and
# it never hurt on any evaluation window.
#
# Standard library only, no network, fully deterministic: two runs on the
# same case write byte-identical output.
#
# Run locally with:
#   MAKEFUN_INPUT_DIR=<dir with *.in> MAKEFUN_OUTPUT_DIR=<out dir> python3 main.py

import json
import math
import os
import statistics
import sys

# Trailing-median window (minutes) defining the recent level.
MED_WINDOW = 21
# Window (minutes) for the robust 1-minute-change scale.
SCALE_WINDOW = 240
# Band floor as a fraction of the last price, and the MAD multiple.
BAND_FLOOR_FRAC = 0.002
BAND_MAD_MULT = 80.0

# The price grid's 5 decimals, and the scorer's (0, 10,000,000) bounds.
MIN_PRICE = 0.00001
MAX_PRICE = 9_999_999.99999


def price5(x):
    """x kept on the grid's 5 decimals and inside the scorer's bounds."""
    return round(min(MAX_PRICE, max(MIN_PRICE, x)) * 1e5) / 1e5


def median_last(prices, n):
    """Median of the last `n` closes (all of them when fewer are held)."""
    window = prices[-n:] if len(prices) >= n else prices[:]
    return statistics.median(window)


def mad_scale(prices, window):
    """Robust scale of 1-minute changes: 1.4826 * median |change|.

    Returns 0 when there is nothing to learn from (a flat or 1-price
    history); the band floor then carries the whole band.
    """
    if len(prices) < 2:
        return 0.0
    start = max(1, len(prices) - window)
    changes = [abs(prices[t] - prices[t - 1]) for t in range(start, len(prices))]
    if not changes:
        return 0.0
    return 1.4826 * statistics.median(changes)


def forecast(prices):
    """The forecast price for one case's price series."""
    if not prices:
        raise ValueError("market has no prices")
    last = prices[-1]
    if not isinstance(last, (int, float)) or not math.isfinite(last):
        raise ValueError("market's last price is not a finite number")
    level = median_last(prices, MED_WINDOW)
    band = max(BAND_FLOOR_FRAC * last, BAND_MAD_MULT * mad_scale(prices, SCALE_WINDOW))
    guess = min(max(last, level - band), level + band)
    return price5(guess)


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(output_dir, exist_ok=True)
    names = sorted(f for f in os.listdir(input_dir) if f.endswith(".in"))
    for name in names:
        case_id = name[: -len(".in")]
        with open(os.path.join(input_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        markets = case.get("markets")
        if not isinstance(markets, list) or len(markets) != 1:
            raise ValueError(f"{name} does not hold exactly one market")
        prices = list(markets[0]["prices"])
        out = {"forecasts": [{"case": case_id, "price": forecast(prices)}]}
        with open(os.path.join(output_dir, case_id + ".out"), "w", encoding="utf-8") as fh:
            fh.write(json.dumps(out, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
