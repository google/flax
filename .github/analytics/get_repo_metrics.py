# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from datetime import datetime
from typing import Callable, List

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import requests
from absl import app, flags

token = os.environ['GITHUB_TOKEN']
endpoint = r'https://api.github.com/graphql'
headers = {'Authorization': f'bearer {token}'}

# ------------------------------------------------------------------------------
# GraphQL
# ------------------------------------------------------------------------------
# NOTE: This GraphQL logic was ported and adapted from this script:
# https://github.com/scientific-python/devstats-data/blob/4c022961abc4ca6061f8719d9c3387e98734b90c/query.py
# It contains style differences from Google's style guide.


def load_query_from_file(fname, repo_owner, repo_name) -> str:
  with open(fname) as fh:
    query = fh.read()
    # Set target repo from template
    query = query.replace('_REPO_OWNER_', repo_owner)
    query = query.replace('_REPO_NAME_', repo_name)
  return query


def send_query(query, query_type, cursor=None):
  """
  Sends a GraphQL to the GitHub API.

  No validation is done on the query before sending. GitHub GraphQL is
  supported with the `cursor` argument.

  Parameters
  ----------
  query : str
    The GraphQL query to be sent
  query_type : {"issues", "pullRequests"}
    The object being queried according to the GitHub GraphQL schema.
    Currently only issues and pullRequests are supported
  cursor : str, optional
    If given, then the cursor is injected into the query to support
    GitHub's GraphQL pagination.

  Returns
  -------
  dict
    The result of the query (json) parsed by `json.loads`

  Notes
  -----
  This is intended mostly for internal use within `get_all_responses`.
  """
  # TODO: Expand this, either by parsing the query type from the query
  # directly or manually adding more query_types to the set
  if query_type not in {'issues', 'pullRequests'}:
    raise ValueError(
      "Only 'issues' and 'pullRequests' queries are currently supported"
    )
  # TODO: Generalize this
  # WARNING: The cursor injection depends on the specific structure of the
  # query, this is the main reason why query types are limited to issues/PRs
  if cursor is not None:
    cursor_insertion_key = query_type + '('
    cursor_ind = query.find(cursor_insertion_key) + len(cursor_insertion_key)
    query = query[:cursor_ind] + f'after:"{cursor}", ' + query[cursor_ind:]
  # Build request payload
  payload = {'query': query}
  response = requests.post(endpoint, json=payload, headers=headers)
  return json.loads(response.content)


def get_all_responses(query, query_type):
  'Helper function to bypass GitHub GraphQL API node limit.'
  # Get data from a single response
  initial_data = send_query(query, query_type)
  data, last_cursor, total_count = parse_single_query(initial_data, query_type)
  print(f'Retrieving {len(data)} out of {total_count} values...')
  # Continue requesting data (with pagination) until all are acquired
  while len(data) < total_count:
    rdata = send_query(query, query_type, cursor=last_cursor)
    pdata, last_cursor, _ = parse_single_query(rdata, query_type)
    data.extend(pdata)
    print(f'Retrieving {len(data)} out of {total_count} values...')
  print('Done.')
  return data


def parse_single_query(data, query_type):
  """
  Parses the data returned by `send_query`

  .. warning::

    Like `send_query`, the logic here depends on the specific structure
    of the query (e.g. it must be an issue or PR query, and must have a
    total count).
  """
  try:
    total_count = data['data']['repository'][query_type]['totalCount']
    data = data['data']['repository'][query_type]['edges']
    last_cursor = data[-1]['cursor']
  except KeyError as e:
    print(data)
    raise e
  return data, last_cursor, total_count


class GithubGrabber:
  """
  Pulls down data via the GitHub APIv.4 given a valid GraphQL query.
  """

  def __init__(self, query_fname, query_type, repo_owner, repo_name):
    """
    Create an object to send/recv queries related to the issue tracker
    for the given repository via the GitHub API v.4.

    The repository to query against is given by:
    https://github.com/<repo_owner>/<repo_name>

    Parameters
    ----------
    query_fname : str
      Path to a valid GraphQL query conforming to the GitHub GraphQL
      schema
    query_type : {"issues", "pullRequests"}
      Type of object that is being queried according to the GitHub GraphQL
      schema. Currently only "issues" and "pullRequests" are supported.
    repo_owner : str
      Repository owner.
    repo_name : str
      Repository name.
    """
    self.query_fname = query_fname
    self.query_type = query_type  # TODO: Parse this directly from query
    self.repo_owner = repo_owner
    self.repo_name = repo_name
    self.raw_data = None
    self.load_query()

  def load_query(self):
    self.query = load_query_from_file(
      self.query_fname, self.repo_owner, self.repo_name
    )

  def get(self):
    self.raw_data = get_all_responses(self.query, self.query_type)


# ------------------------------------------------------------------------------
# metrics helpers
# ------------------------------------------------------------------------------


def _to_datetime(date_str: str) -> datetime:
  return datetime.fromisoformat(date_str.replace('Z', ''))


def _get_issues_features(issues):
  for issue in issues:
    issue = issue['node']

    created_at = _to_datetime(issue['createdAt'])
    time_labeled_or_converted = None
    time_issue_closed = None

    for event in issue['timelineItems']['edges']:
      event = event['node']

      if event['__typename'] in {'LabeledEvent', 'ConvertedToDiscussionEvent'}:
        time_labeled_or_converted = _to_datetime(event['createdAt'])

      if event['__typename'] == 'ClosedEvent':
        time_issue_closed = _to_datetime(event['createdAt'])

    yield {
      'created_at': created_at,
      'time_labeled_or_converted': time_labeled_or_converted,
      'time_issue_closed': time_issue_closed,
      'issue_closed': issue['state'] == 'CLOSED',
    }


def _get_pr_features(prs):
  for pr in prs:
    pr = pr['node']

    created_at = _to_datetime(pr['createdAt'])
    ready_for_review_at = _to_datetime(pr['createdAt'])
    time_labeled_or_assigned = None
    time_merged_or_closed = None
    time_review = None

    if pr['reviews']['nodes']:
      review = pr['reviews']['nodes'][0]
      time_review = _to_datetime(review['createdAt'])

    for event in pr['timelineItems']['edges']:
      event = event['node']

      if (
        time_labeled_or_assigned is None
        and event['__typename'] == 'LabeledEvent'
        and 'cla:' not in event['label']['name']
      ):
        time_labeled_or_assigned = _to_datetime(event['createdAt'])

      if (
        time_labeled_or_assigned is None
        and event['__typename'] == 'AssignedEvent'
      ):
        time_labeled_or_assigned = _to_datetime(event['createdAt'])

      if event['__typename'] in {'ClosedEvent', 'MergedEvent'}:
        time_merged_or_closed = _to_datetime(event['createdAt'])

      if event['__typename'] == 'ReadyForReviewEvent':
        ready_for_review_at = _to_datetime(event['createdAt'])

    yield {
      'created_at': created_at,
      'ready_for_review_at': ready_for_review_at,
      'time_labeled_or_assigned': time_labeled_or_assigned,
      'time_merged_or_closed': time_merged_or_closed,
      'time_review': time_review,
      'pr_closed': pr['state'] != 'OPEN',
    }


def _start_of_month(date: datetime) -> datetime:
  return date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def _shift_n_months(date: datetime, n: int) -> datetime:
  month = ((date.month + n - 1) % 12) + 1

  # shift to next year if necessary
  if date.month > month:
    date = date.replace(year=date.year + 1)

  date = date.replace(month=month)

  return date


def _rolling_window(
  df: pd.DataFrame,
  f: Callable[[pd.DataFrame], pd.Series],
  window_size: int = 6,
  step: int = 1,
) -> pd.DataFrame:
  # start of month of the first issue
  start: datetime = df.iloc[0]['created_at'].replace(
    day=1, hour=0, minute=0, second=0, microsecond=0
  )
  end = _shift_n_months(start, window_size)

  last_month = _start_of_month(df.iloc[-1]['created_at'])
  last_month = _shift_n_months(last_month, 1)

  rows: List[pd.Series] = []
  while end < last_month:
    row = f(df[(df['created_at'] >= start) & (df['created_at'] < end)])
    row['period_start'] = start
    row['period_end'] = end
    rows.append(row)
    start = _shift_n_months(start, step)
    end = _shift_n_months(end, step)

  df = pd.DataFrame(rows)
  df = df[['period_start', 'period_end'] + list(df.columns[:-2])]

  return df


def _process_prs(df: pd.DataFrame) -> pd.Series:
  return pd.Series(
    {
      'pr_response_time': df['pr_response_time'].dt.days.mean(),
      'pr_resolution_time': df['pr_resolution_time'].dt.days.mean(),
    }
  )


def _process_issues(df: pd.DataFrame) -> pd.Series:
  return pd.Series(
    {
      'issue_response_time': df['issue_response_time'].dt.days.mean(),
      'issue_resolution_time': df['issue_resolution_time'].dt.days.mean(),
    }
  )


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
FLAGS = flags.FLAGS
flags.DEFINE_string('repo_owner', 'google', 'User name or organization')
flags.DEFINE_string('repo_name', 'flax', 'Name of the repository')


def main(_):
  repo_owner: str = FLAGS.repo_owner
  repo_name: str = FLAGS.repo_name

  # Download issue data
  issues = GithubGrabber(
    '.github/analytics/issue_activity_since_date.gql',
    'issues',
    repo_owner=repo_owner,
    repo_name=repo_name,
  )
  issues.get()

  df_issues = df_issues0 = pd.DataFrame(
    list(_get_issues_features(issues.raw_data))
  )
  df_issues['issue_response_time'] = (
    df_issues['time_labeled_or_converted'] - df_issues['created_at']
  )
  df_issues['issue_resolution_time'] = (
    df_issues['time_issue_closed'] - df_issues['created_at']
  )

  df_issues = _rolling_window(df_issues, _process_issues)

  prs = GithubGrabber(
    '.github/analytics/pr_data_query.gql',
    'pullRequests',
    repo_owner=repo_owner,
    repo_name=repo_name,
  )
  prs.get()

  df_prs = df_prs0 = pd.DataFrame(list(_get_pr_features(prs.raw_data)))
  time_response = df_prs[['time_labeled_or_assigned', 'time_review']].min(
    axis=1
  )
  df_prs['pr_response_time'] = time_response - df_prs['ready_for_review_at']
  df_prs['pr_resolution_time'] = (
    df_prs['time_merged_or_closed'] - df_prs['ready_for_review_at']
  )

  df_prs = _rolling_window(df_prs, _process_prs)

  # get cummulative issues
  df_issues0 = df_issues0.copy()
  df_issues0['number_of_issues'] = 1
  df_issues0['number_of_issues'] = df_issues0['number_of_issues'].cumsum()

  # get cummulative prs
  df_prs0 = df_prs0.copy()
  df_prs0['number_of_prs'] = 1
  df_prs0['number_of_prs'] = df_prs0['number_of_prs'].cumsum()

  # plot cumulative issues
  plt.figure()
  plt.plot(df_issues0['created_at'], df_issues0['number_of_issues'])
  plt.xlabel('Date')
  plt.ylabel('Number of issues')
  plt.title('Number of issues')
  plt.gca().xaxis.set_major_locator(plt.MaxNLocator(5))
  plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

  # plot cumulative prs
  plt.figure()
  plt.plot(df_prs0['created_at'], df_prs0['number_of_prs'])
  plt.xlabel('Date')
  plt.ylabel('Number of PRs')
  plt.title('Number of PRs')
  plt.gca().xaxis.set_major_locator(plt.MaxNLocator(5))
  plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

  # plot for isssue_response_time
  plt.figure()
  plt.plot(df_issues['period_end'], df_issues['issue_response_time'])
  plt.xlabel('Date')
  plt.ylabel('Issue Response Time (days)')
  plt.title('Issue Response Time')
  plt.gca().xaxis.set_major_locator(plt.MaxNLocator(5))
  plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
  plt.ylim(0)

  # plot for issue_resolution_time
  plt.figure()
  plt.plot(df_issues['period_end'], df_issues['issue_resolution_time'])
  plt.xlabel('Date')
  plt.ylabel('Issue Resolution Time (days)')
  plt.title('Issue Resolution Time')
  plt.gca().xaxis.set_major_locator(plt.MaxNLocator(5))
  plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
  plt.ylim(0)

  # plot for pr_response_time
  plt.figure()
  plt.plot(df_prs['period_end'], df_prs['pr_response_time'])
  plt.xlabel('Date')
  plt.ylabel('Pull Request Response Time (days)')
  plt.title('Pull Request Response Time')
  plt.gca().xaxis.set_major_locator(plt.MaxNLocator(5))
  plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
  plt.ylim(0)

  # plot for pr_resolution_time
  plt.figure()
  plt.plot(df_prs['period_end'], df_prs['pr_resolution_time'])
  plt.xlabel('Date')
  plt.ylabel('Pull Request Resolution Time (days)')
  plt.title('Pull Request Resolution Time')
  plt.gca().xaxis.set_major_locator(plt.MaxNLocator(5))
  plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
  plt.ylim(0)

  # show plots
  plt.show()


if __name__ == '__main__':
  app.run(main)
