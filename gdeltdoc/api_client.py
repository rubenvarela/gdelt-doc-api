import requests
import pandas as pd
import datetime
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from gdeltdoc.filters import Filters

from typing import Dict, List, Optional, Union

from gdeltdoc.helpers import load_json

from gdeltdoc._version import version


class GdeltDoc:
    """
    API client for the GDELT 2.0 Doc API

    ```
    from gdeltdoc import GdeltDoc, Filters

    f = Filters(
        keyword = "climate change",
        start_date = "2020-05-10",
        end_date = "2020-05-11"
    )

    gd = GdeltDoc()

    # Search for articles matching the filters
    articles = gd.article_search(f)

    # Get a timeline of the number of articles matching the filters
    timeline = gd.timeline_search("timelinevol", f)
    ```

    ### Article List
    The article list mode of the API generates a list of news articles that match the filters.
    The client returns this as a pandas DataFrame with columns `url`, `url_mobile`, `title`,
    `seendate`, `socialimage`, `domain`, `language`, `sourcecountry`.

    ### Timeline Search
    There are 5 available modes when making a timeline search:
    * `timelinevol` - a timeline of the volume of news coverage matching the filters,
        represented as a percentage of the total news articles monitored by GDELT.
    * `timelinevolraw` - similar to `timelinevol`, but has the actual number of articles
        and a total rather than a percentage
    * `timelinelang` - similar to `timelinevol` but breaks the total articles down by published language.
        Each language is returned as a separate column in the DataFrame.
    * `timelinesourcecountry` - similar to `timelinevol` but breaks the total articles down by the country
        they were published in. Each country is returned as a separate column in the DataFrame.
    * `timelinetone` - a timeline of the average tone of the news coverage matching the filters.
        See [GDELT's documentation](https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/)
        for more information about the tone metric.
    """

    def __init__(self, json_parsing_max_depth: int = 100, time_slice_minutes: int = 60, max_workers: Optional[int] = None) -> None:
        """
        Params
        ------
        json_parsing_max_depth
            A parameter for the json parsing function that removes illegal character. If 100 it will remove at max
            100 characters before exiting with an exception
        time_slice_minutes
            The size of time slices in minutes when fetching articles (minimum 30)
        max_workers
            Maximum number of workers for parallel execution. If None, uses ThreadPoolExecutor's default
        """
        self.max_depth_json_parsing = json_parsing_max_depth
        self.time_slice_minutes = max(30, time_slice_minutes)  # Minimum 30 minutes
        self.max_workers = max_workers

    def article_search(self, filters: Filters, time_slice_minutes: Optional[int] = None) -> pd.DataFrame:
        """
        Make a query against the `ArtList` API to return a DataFrame of news articles that
        match the supplied filters. The date range will be automatically split into chunks
        to overcome the 250 article limit per query.

        Params
        ------
        filters
            A `gdelt-doc.Filters` object containing the filter parameters for this query.
        time_slice_minutes
            The time slices in minutes. Overrides the instance setting if provided.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame of the articles returned from the API.
        """
        # Use instance default if not provided
        slice_minutes = self.time_slice_minutes if time_slice_minutes is None else max(30, time_slice_minutes)

        # Always use time slicing
        return self._article_search_with_time_slicing(filters, slice_minutes)

    def timeline_search(self, mode: str, filters: Filters) -> pd.DataFrame:
        """
        Make a query using one of the API's timeline modes.

        Params
        ------
        mode
            The API mode to call. Must be one of "timelinevol", "timelinevolraw",
            "timelinetone", "timelinelang", "timelinesourcecountry".

            See https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/ for a
            longer description of each mode.

        filters
            A `gdelt-doc.Filters` object containing the filter parameters for this query.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame of the articles returned from the API.
        """
        timeline = self._query(mode, filters.query_string)

        # If no results
        if len(timeline["timeline"]) == 0:
            return pd.DataFrame()

        results = {
            "datetime": [entry["date"] for entry in timeline["timeline"][0]["data"]]
        }

        for series in timeline["timeline"]:
            results[series["series"]] = [entry["value"] for entry in series["data"]]

        if mode == "timelinevolraw":
            results["All Articles"] = [
                entry["norm"] for entry in timeline["timeline"][0]["data"]
            ]

        formatted = pd.DataFrame(results)
        formatted["datetime"] = pd.to_datetime(formatted["datetime"])

        return formatted

    def _query(self, mode: str, query_string: str) -> Dict:
        """
        Submit a query to the GDELT API and return the results as a parsed JSON object.

        Params
        ------
        mode
            The API mode to call. Must be one of "artlist", "timelinevol",
            "timelinevolraw", "timelinetone", "timelinelang", "timelinesourcecountry".

        query_string
            The query parameters and date range to call the API with.

        Returns
        -------
        Dict
            The parsed JSON response from the API.
        """
        if mode not in [
            "artlist",
            "timelinevol",
            "timelinevolraw",
            "timelinetone",
            "timelinelang",
            "timelinesourcecountry",
        ]:
            raise ValueError(f"Mode {mode} not in supported API modes")

        headers = {
            "User-Agent": f"GDELT DOC Python API client {version} - https://github.com/alex9smith/gdelt-doc-api"
        }

        response = requests.get(
            f"https://api.gdeltproject.org/api/v2/doc/doc?query={query_string}&mode={mode}&format=json",
            headers=headers,
        )

        if response.status_code not in [200, 202]:
            raise ValueError(
                "The gdelt api returned a non-successful statuscode. This is the response message: {}".format(
                    response.text
                )
            )

        # Response is text/html if it's an error and application/json if it's ok
        if "text/html" in response.headers["content-type"]:
            raise ValueError(
                f"The query was not valid. The API error message was: {response.text.strip()}"
            )

        return load_json(response.content, self.max_depth_json_parsing)

    def _article_search_with_time_slicing(self, filters: Filters, time_slice_minutes: int = 60) -> pd.DataFrame:
        """
        Perform an article search by breaking the date range into smaller chunks.
        This helps overcome the 250 article limit per query.

        Params
        ------
        filters
            A `gdelt-doc.Filters` object containing the filter parameters for this query.
        time_slice_minutes
            The size of each time slice in minutes.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame combining articles from all time slices with duplicates removed.
        """
        # Extract the start and end dates from the query string
        query_string = filters.query_string
        start_param = "&startdatetime="
        end_param = "&enddatetime="
        max_param = "&maxrecords="

        # If timespan is used instead of start/end dates, fall back to regular query
        if "&timespan=" in query_string:
            return self._query_to_dataframe("artlist", query_string)

        # Find positions of datetime parameters
        start_pos = query_string.find(start_param)
        if start_pos == -1:
            # If start/end datetime parameters not found, fall back to regular query
            return self._query_to_dataframe("artlist", query_string)

        start_pos += len(start_param)
        end_pos = query_string.find(end_param) + len(end_param)
        max_pos = query_string.find(max_param)

        # Extract date strings
        start_date_str = query_string[start_pos:query_string.find("&", start_pos)]
        end_date_str = query_string[end_pos:query_string.find("&", end_pos)]

        # Convert to datetime objects for slicing
        try:
            start_date = datetime.datetime.strptime(start_date_str, "%Y%m%d%H%M%S")
            end_date = datetime.datetime.strptime(end_date_str, "%Y%m%d%H%M%S")
        except ValueError:
            # If there's an issue parsing dates, try to recover if possible
            try:
                # Try alternative formats
                if len(start_date_str) == 8:  # YYYYMMDD
                    start_date = datetime.datetime.strptime(start_date_str, "%Y%m%d")
                    end_date = datetime.datetime.strptime(end_date_str, "%Y%m%d")
                else:
                    # If we can't parse the dates, fall back to regular query
                    return self._query_to_dataframe("artlist", query_string)
            except ValueError:
                # If still can't parse, fall back to regular query
                return self._query_to_dataframe("artlist", query_string)

        # Check if date range is too small for slicing
        time_diff = (end_date - start_date).total_seconds() / 60  # Time difference in minutes
        if time_diff <= time_slice_minutes:
            # If time range is smaller than slice size, use regular query
            return self._query_to_dataframe("artlist", query_string)

        # Generate time slices
        date_range = [date.strftime("%Y%m%d%H%M%S") for date in
                     pd.date_range(start_date, end_date, freq=f"{time_slice_minutes}min")]

        if len(date_range) <= 1:
            # If there's only one slice or empty range, use the original query
            return self._query_to_dataframe("artlist", query_string)

        # Create query strings for each time slice
        query_strings = []
        base_query = query_string[:start_pos]
        max_records_part = query_string[max_pos:]

        for i in range(len(date_range) - 1):
            tmp_start_date, tmp_end_date = date_range[i], date_range[i + 1]
            # Construct new query string with the time slice
            slice_query = f"{base_query}{tmp_start_date}{end_param}{tmp_end_date}{max_records_part}"
            query_strings.append(slice_query)

        # Execute queries in parallel
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            query_func = partial(self._query, "artlist")
            # Use a try/except to catch any errors during parallel execution
            try:
                results = list(executor.map(query_func, query_strings))
            except Exception as e:
                # If parallel execution fails, try sequential execution as fallback
                try:
                    results = [query_func(qs) for qs in query_strings]
                except Exception:
                    # If both parallel and sequential fail, return an empty DataFrame
                    return pd.DataFrame()

        # Process and combine results
        dataframes = []
        for result in results:
            if result and "articles" in result and result["articles"]:
                dataframes.append(pd.DataFrame(result["articles"]))

        if not dataframes:
            return pd.DataFrame()

        # Combine all results and remove duplicates
        combined_df = pd.concat(dataframes, ignore_index=True)
        return combined_df.drop_duplicates(subset=["url"]).reset_index(drop=True)

    def _query_to_dataframe(self, mode: str, query_string: str) -> pd.DataFrame:
        """
        Helper method to query the API and convert the results to a DataFrame.

        Params
        ------
        mode
            The API mode to call.
        query_string
            The query parameters to use.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame of the results.
        """
        result = self._query(mode, query_string)
        if mode == "artlist" and "articles" in result:
            return pd.DataFrame(result["articles"])
        return pd.DataFrame()
