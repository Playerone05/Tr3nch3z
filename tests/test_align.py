from signalbot.align import assign_trading_day, normalize_headline, story_key, to_utc

# Mon Oct 12 – Fri Oct 16 2026 plus Mon Oct 19 (weekend skipped)
DAYS = ["2026-10-12", "2026-10-13", "2026-10-14", "2026-10-15", "2026-10-16", "2026-10-19"]


def day(ts):
    return assign_trading_day(ts, DAYS, "America/New_York", "16:00")


def test_news_before_close_counts_same_day():
    # 3:59 PM ET Tuesday = 19:59 UTC (EDT is UTC-4)
    assert day("2026-10-13T19:59:00Z") == "2026-10-13"


def test_news_at_or_after_close_rolls_to_next_day():
    assert day("2026-10-13T20:00:00Z") == "2026-10-14"   # exactly 4:00 PM ET
    assert day("2026-10-13T23:30:00Z") == "2026-10-14"   # after hours


def test_overnight_news_counts_for_that_morning():
    # 2 AM ET Wednesday
    assert day("2026-10-14T06:00:00Z") == "2026-10-14"


def test_friday_after_close_and_weekend_roll_to_monday():
    assert day("2026-10-16T21:00:00Z") == "2026-10-19"
    assert day("2026-10-17T15:00:00Z") == "2026-10-19"   # Saturday
    assert day("2026-10-18T22:00:00Z") == "2026-10-19"   # Sunday night


def test_holiday_skipped_when_calendar_knows_it():
    days = ["2026-11-25", "2026-11-27"]                  # Thanksgiving (26th) closed
    assert assign_trading_day("2026-11-25T22:00:00Z", days) == "2026-11-27"


def test_news_past_known_calendar_falls_back_to_next_weekday():
    # Friday after close, calendar ends Friday -> Monday
    assert assign_trading_day("2026-10-23T21:00:00Z", DAYS) == "2026-10-26"


def test_news_before_known_calendar_does_not_jump_to_first_day():
    assert assign_trading_day("2018-06-05T14:00:00Z", DAYS) == "2018-06-05"


def test_unix_seconds_accepted():
    assert to_utc(1760385540).isoformat() == "2025-10-13T19:59:00+00:00"


def test_story_key_ignores_case_and_punctuation():
    a = "NVIDIA Beats Estimates; Shares Jump 5%"
    b = "nvidia beats estimates -- shares jump 5% "
    assert normalize_headline(a) == "nvidia beats estimates shares jump 5"
    assert story_key(a) == story_key(b)
    assert story_key(a) != story_key("NVIDIA misses estimates")
