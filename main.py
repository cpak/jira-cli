#! /usr/bin/env python3

import argparse
from dataclasses import dataclass
import json
import logging
import os
import re
import subprocess
import requests
from typing import Any, Callable, Dict, List, Optional, Union, cast


logger = logging.getLogger("jira")
logger.setLevel(logging.DEBUG)

_CACHE: Dict[str, Optional["Config"]] = {
    "config": None,
}


@dataclass
class Config:
    token: str
    user: str
    project_key: str
    base_url: str
    todo_status: str
    in_progress_id: str
    done_id: str


@dataclass
class Issue:
    key: str
    issue_type: str
    summary: str
    status: str
    created: Optional[str] = None
    parent_key: Optional[str] = None
    assignee: Optional[str] = None
    raw: Optional[Dict] = None

    def is_bug(self) -> bool:
        return self.issue_type == "Bug"


@dataclass
class Transition:
    id: str
    name: str


def _load_config() -> Config:
    if _CACHE["config"] is None:
        config_path = os.path.expanduser("~/.jira")
        if not os.path.exists(config_path):
            raise Exception(f"Config file not found: {config_path}")
        with open(config_path) as f:
            c = json.load(f)
            _CACHE["config"] = Config(
                c["token"],
                c["user"],
                c["project_key"],
                c["base_url"],
                c["todo_status"],
                c["in_progress_id"],
                c["done_id"],
            )
    return _CACHE["config"]


def safe_get(
    path: "Union[str, List[str]]",
    obj: object,
    default=None,
) -> Any:
    if type(path) is str:
        path = [path]
    c = obj
    for k in path:
        if isinstance(c, dict):
            if k in c:
                c = c[k]
                continue
        elif isinstance(c, list):
            try:
                i = int(k)
                c = cast(List[Any], c)[i]
                continue
            except (ValueError, IndexError):
                return default
        else:
            if c and hasattr(c, k):
                c = getattr(c, k)
                continue
        return default
    return c


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {_load_config().token}",
        "Accept": "application/json",
    }


def _api_url(path: str) -> str:
    base_url = _load_config().base_url + "/rest/api/2"
    return f"{base_url}/{path}"


def _api_get(url, params={}):
    response = requests.get(url, params=params, headers=_headers())
    return response.json()


def _parse_issue(raw: dict) -> Issue:
    created = safe_get(["fields", "created"], raw)
    return Issue(
        safe_get(["key"], raw),
        safe_get(["fields", "issuetype", "name"], raw),
        safe_get(["fields", "summary"], raw),
        safe_get(["fields", "status", "name"], raw),
        created.split("T")[0] if created else None,
        safe_get(["fields", "parent", "key"], raw),
        safe_get(["fields", "assignee", "displayName"], raw),
        raw,
    )


def _parse_issues(body: dict) -> list:
    return [_parse_issue(issue) for issue in body["issues"]]


def _print_table(rows: list) -> None:
    if len(rows) == 0:
        return
    widths = [max(map(len, col)) for col in zip(*rows)]
    for row in rows:
        print("  ".join((val.ljust(width) for val, width in zip(row, widths))))


def _issues_to_rows(issues: list) -> list:
    return [
        [
            issue.key,
            issue.issue_type,
            issue.status,
            issue.summary,
            issue.created,
            issue.assignee or "",
        ]
        for issue in issues
    ]


def _slugify(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^a-z0-9-]", "", s)
    return s


def _issue(issue_key: str) -> Issue:
    issue_url = _api_url(f"issue/{issue_key}")
    body = _api_get(issue_url)
    return _parse_issue(body)


def _issue_key_or_select(issue_key: Optional[str], cmd: List[str]) -> Optional[str]:
    if issue_key:
        return issue_key
    else:
        issue = _select_issue(cmd)
        issue_key = issue.key if issue else None
    return issue_key


def _issue_browse_url(issue: Union[str, Issue]) -> str:
    issue_key = issue if isinstance(issue, str) else issue.key
    return f"{_load_config().base_url}/browse/{issue_key}"


def _search(query: str) -> list:
    search_url = _api_url("search")
    params = {"jql": query}
    body = _api_get(search_url, params=params)
    return _parse_issues(body)


def _todo() -> list:
    cfg = _load_config()
    return _search(f'project = {cfg.project_key} AND status = "{cfg.todo_status}"')


def _mine():
    project_key = _load_config().project_key
    return _search(
        f"project = {project_key} AND assignee = currentUser() AND statusCategory != Done"
    )


def _branch_name(issue: Union[str, Issue], prefix: Optional[str] = None) -> str:
    if isinstance(issue, str):
        issue = _issue(issue)
    prefix = prefix or ("fix" if issue.is_bug() else "feat")
    branch_name = f"{prefix}/{issue.key}-{_slugify(issue.summary)}"
    branch_name = branch_name[:80]
    return branch_name


def _branch(branch_name: str):
    confirmation = input(f"Create branch {branch_name}? (Y/n): ").lower()
    if not confirmation or confirmation == "y":
        subprocess.run(["git", "checkout", "-b", branch_name], check=True)


def _select_issue(list_issues_cmd: List[str]) -> Optional[Issue]:
    list_issues = subprocess.Popen(list_issues_cmd, stdout=subprocess.PIPE, text=True)
    fzf = subprocess.Popen(
        ["fzf"], stdin=list_issues.stdout, stdout=subprocess.PIPE, text=True
    )
    output, error = fzf.communicate()
    if error:
        raise Exception(error)
    issue_key = output.split(" ")[0] if output else None
    if issue_key:
        return _issue(issue_key)


def _select_transition(issue_key: str) -> Optional[Transition]:
    list_transitions = subprocess.Popen(
        ["jira", "transitions", issue_key], stdout=subprocess.PIPE, text=True
    )
    fzf = subprocess.Popen(
        ["fzf"], stdin=list_transitions.stdout, stdout=subprocess.PIPE, text=True
    )
    output, error = fzf.communicate()
    if error:
        raise Exception(error)
    transition_id, transition_name = (
        output.strip().split(" ", maxsplit=1) if output else (None, None)
    )
    if transition_id and transition_name:
        return Transition(transition_id, transition_name)


def _assign(issue_key: str, user: str):
    issue_url = _api_url(f"issue/{issue_key}/assignee")
    body = {"name": user}
    response = requests.put(issue_url, json=body, headers=_headers())
    response.raise_for_status()


def _move(issue_key: str, target_status: str):
    issue_url = _api_url(f"issue/{issue_key}/transitions")
    body = {"transition": {"id": target_status}}
    response = requests.post(issue_url, json=body, headers=_headers())
    response.raise_for_status()


def _transitions(issue_key: str) -> List[Transition]:
    issue_url = _api_url(f"issue/{issue_key}/transitions")
    response = requests.get(issue_url, headers=_headers())
    response.raise_for_status()
    ts = safe_get("transitions", response.json(), [])
    return [Transition(t["id"], t["name"]) for t in ts]


def _create_issue(
    issue_type: str, summary: str, parent_key: Optional[str] = None
) -> Issue:
    cfg = _load_config()
    issue_url = _api_url("issue")
    body = {
        "fields": {
            "project": {"key": cfg.project_key},
            "summary": summary,
            "issuetype": {"name": issue_type},
        }
    }
    if parent_key:
        body["fields"]["parent"] = {"key": parent_key}
        body["fields"]["issuetype"] = {"name": "Sub-task"}
    response = requests.post(issue_url, json=body, headers=_headers())
    response.raise_for_status()
    return _parse_issue(response.json())


def search(args: argparse.Namespace):
    query = args.query
    _print_table(_issues_to_rows(_search(query)))


def todo(_):
    _print_table(_issues_to_rows(_todo()))


def mine(_):
    issues = _mine()
    _print_table(_issues_to_rows(issues))


def branch(args: argparse.Namespace):
    issue_key = _issue_key_or_select(args.issue, ["jira", "mine"])
    prefix: Optional[str] = safe_get("prefix", args)
    if issue_key:
        _branch(_branch_name(issue_key, prefix))


def move(args: argparse.Namespace):
    issue_key = _issue_key_or_select(args.issue, ["jira", "mine"])
    status = safe_get("status", args)
    if issue_key and not status:
        status = _select_transition(issue_key)
    if issue_key and status:
        _move(issue_key, status.id)
        print(f"{issue_key} -> {status.name}")


def create(args: argparse.Namespace):
    issue = _create_issue(args.type, args.summary, args.parent)
    print(f"Created {issue.key} {issue.summary}")


def work(args: argparse.Namespace):
    issue_key = _issue_key_or_select(args.issue, ["jira", "todo"])
    if not issue_key:
        return
    cfg = _load_config()
    _move(issue_key, cfg.in_progress_id)
    _assign(issue_key, cfg.user)
    issue = _issue(issue_key)
    print(f"{issue.key} {issue.summary} -> {issue.status} {issue.assignee}")


def done(args: argparse.Namespace):
    issue_key = _issue_key_or_select(args.issue, ["jira", "mine"])
    if not issue_key:
        return
    cfg = _load_config()
    _move(issue_key, cfg.done_id)
    issue = _issue(issue_key)
    print(f"{issue.key} {issue.summary} -> {issue.status} {issue.assignee}")


def dump_issue(args: argparse.Namespace):
    cmd = safe_get("list_cmd", args, "todo")
    issue = _select_issue(["jira", cmd])
    if not issue:
        return
    print(json.dumps(issue.raw, indent=2))


def open_issue(args: argparse.Namespace):
    cmd = safe_get("list_cmd", args) or "todo"
    issue_key = _issue_key_or_select(args.issue, ["jira", cmd])
    if not issue_key:
        return
    issue_url = _issue_browse_url(issue_key)
    browser = os.environ.get("BROWSER", "open")
    subprocess.run([browser, issue_url], check=True)


def url(args: argparse.Namespace):
    cmd = safe_get("list_cmd", args, "mine")
    issue_key = _issue_key_or_select(args.issue, ["jira", cmd])
    if not issue_key:
        return
    print(_issue_browse_url(issue_key))


def transitions(args: argparse.Namespace):
    issue_key = args.issue
    for t in _transitions(issue_key):
        print(f"{t.id} {t.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="jira", description="jira cli")

    subparsers = parser.add_subparsers(dest="command")

    # branch
    branch_parser = subparsers.add_parser(
        "branch", help="create a branch from an issue"
    )
    branch_parser.add_argument("-i", "--issue", required=False, help="issue key")
    branch_parser.add_argument("-p", "--prefix", required=False, help="branch prefix")
    branch_parser.set_defaults(func=branch)

    # create issue
    create_parser = subparsers.add_parser(
        "create", help="create a create from an issue"
    )
    create_parser.add_argument(
        "-p", "--parent", type=str, required=False, help="parent issue key"
    )
    create_parser.add_argument(
        "-s", "--summary", type=str, required=True, help="issue summary"
    )
    create_parser.add_argument(
        "-t", "--type", type=str, required=False, help="issue type", default="Task"
    )
    create_parser.set_defaults(func=create)

    # dump_issue
    dump_issue_parser = subparsers.add_parser(
        "dump_issue", help="dump the raw json of an issue"
    )
    dump_issue_parser.add_argument("list_cmd", nargs="?", help="list command")
    dump_issue_parser.set_defaults(func=dump_issue)

    # mine
    mine_parser = subparsers.add_parser("mine", help="list my issues")
    mine_parser.set_defaults(func=mine)

    # open
    open_parser = subparsers.add_parser("open", help="open issue in browser")
    open_parser.add_argument("-i", "--issue", required=False, help="issue key")
    open_parser.add_argument("-l", "--list-cmd", required=False, help="list command")
    open_parser.set_defaults(func=open_issue)

    # search
    search_parser = subparsers.add_parser("search", help="search for issues")
    search_parser.add_argument("query", nargs=1, help="jql query")
    search_parser.set_defaults(func=search)

    # todo
    todo_parser = subparsers.add_parser("todo", help="list todo issues")
    todo_parser.set_defaults(func=todo)

    # transitions
    transitions_parser = subparsers.add_parser(
        "transitions", help="list transitions for an issue"
    )
    transitions_parser.add_argument("issue", nargs="?", help="issue key")
    transitions_parser.set_defaults(func=transitions)

    # move
    move_parser = subparsers.add_parser("move", help="move an issue")
    move_parser.add_argument("-i", "--issue", required=False, help="issue key")
    move_parser.add_argument("-s", "--status", required=False, help="status")
    move_parser.set_defaults(func=move)

    # work
    work_parser = subparsers.add_parser("work", help="start working on an issue")
    work_parser.add_argument("issue", nargs="?", help="issue key")
    work_parser.set_defaults(func=work)

    # url
    url_parser = subparsers.add_parser("url", help="get issue url")
    url_parser.add_argument(
        "-l", "--list-cmd", type=str, required=False, help="list command"
    )
    url_parser.add_argument("issue", nargs="?", help="issue key")
    url_parser.set_defaults(func=url)

    # done
    done_parser = subparsers.add_parser("done", help="finish working on an issue")
    done_parser.add_argument("issue", nargs="?", help="issue key")
    done_parser.set_defaults(func=done)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        exit(1)

    # print(args)
    args.func(args)
