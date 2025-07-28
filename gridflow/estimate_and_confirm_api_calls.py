#!/usr/bin/env python3
"""
Rate-limited API call time estimator for Renewables Ninja with configurable limits.
Handles per-second, per-minute, and per-hour rate limits.
"""

import math

def estimate_and_confirm_api_calls(num_geographic_areas, measurement_points_per_area):
    """
    Calculate how long API calls will take and ask user confirmation to proceed.
    """
    MAX_REQUESTS_PER_SECOND = 1 
    MAX_REQUESTS_PER_MINUTE = 6
    MAX_REQUESTS_PER_HOUR = 50

    total_data_points = num_geographic_areas * measurement_points_per_area
    total_api_calls = total_data_points * 2

    def calculate_time_for_queries_within_hour(num_queries):
        """
        Calculate seconds needed for a batch of queries within a single hour.
        Accounts for the 6-per-minute rate limit.
        """
        if num_queries <= 0:
            return 0

        complete_minutes_needed = num_queries // MAX_REQUESTS_PER_MINUTE
        queries_in_final_minute = num_queries % MAX_REQUESTS_PER_MINUTE

        if num_queries <= MAX_REQUESTS_PER_MINUTE:
            time_with_second_limit = num_queries * (1.0 / MAX_REQUESTS_PER_SECOND)
            time_with_minute_limit = num_queries
            return max(time_with_second_limit, time_with_minute_limit)
        elif queries_in_final_minute == 0:
            return (complete_minutes_needed - 1) * 60 + MAX_REQUESTS_PER_MINUTE
        else:
            return complete_minutes_needed * 60 + queries_in_final_minute

    complete_hour_windows = total_api_calls // MAX_REQUESTS_PER_HOUR
    queries_in_final_window = total_api_calls % MAX_REQUESTS_PER_HOUR

    time_per_complete_window = calculate_time_for_queries_within_hour(MAX_REQUESTS_PER_HOUR)
    time_for_final_queries = calculate_time_for_queries_within_hour(queries_in_final_window)

    if complete_hour_windows == 0:
        total_seconds = time_for_final_queries
    else:
        if queries_in_final_window == 0:
            hours_between_windows = (complete_hour_windows - 1) * 3600
            last_window_query_time = time_per_complete_window
            total_seconds = hours_between_windows + last_window_query_time
        else:
            hours_after_complete_windows = complete_hour_windows * 3600
            leftover_query_time = time_for_final_queries
            total_seconds = hours_after_complete_windows + leftover_query_time

    hours = total_seconds // 3600
    minutes_remainder = total_seconds % 3600
    minutes = minutes_remainder // 60
    seconds = minutes_remainder % 60

    if hours > 0:
        readable_time = f"{hours}h {minutes}min {seconds}s"
    elif minutes > 0:
        readable_time = f"{minutes}min {seconds}s"
    else:
        readable_time = f"{seconds}s"

    print(f"""
API Time Estimate:
- Total queries: {total_api_calls} ({total_data_points} points × 2)
- Estimated time: {readable_time}
""")

    while True:
        user_response = input("Do you want to proceed? (y/n): ").strip().lower()

        if user_response in ['yes', 'y']:
            print("Starting data collection now...")
            return True
        elif user_response in ['no', 'n']:
            print("Data collection cancelled.")
            return False
        else:
            print("Please enter 'y' for yes or 'n' for no")
