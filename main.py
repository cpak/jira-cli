#! /usr/bin/env python3

import argparse
from dataclasses import dataclass
import json
import logging
import os
import re
import readline
import subprocess
import requests
from typing import Any, Dict, List, Optional, Tuple, Union, cast
import tempfile


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
    todo_id: str
    in_progress_id: str
    done_id: str
    default_pr_labels: List[str]


@dataclass
class Issue:
    key: str
    issue_type: str
    summary: str
    status: str
    created: Optional[str] = None
    parent_key: Optional[str] = None
    subtask_keys: Optional[List[str]] = None
    assignee: Optional[str] = None
    raw: Optional[Dict] = None

    def __hash__(self) -> int:
        return hash(self.key)

    def is_bug(self) -> bool:
        return self.issue_type == "Bug"

    def is_epic(self) -> bool:
        return self.issue_type == "Epic"


@dataclass
class IssueNode:
    issue: Issue
    children: List["IssueNode"]

    def to_json(self) -> Dict:
        return {
            "issue": {
                "key": self.issue.key,
                "summary": self.issue.summary,
                "status": self.issue.status,
                "assignee": self.issue.assignee,
            },
            "children": [c.to_json() for c in self.children],
        }


IssueFields = List[str]


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
                c["todo_id"],
                c["in_progress_id"],
                c["done_id"],
                safe_get("default_pr_labels", c, []),
            )
    return _CACHE["config"]


def _input(prompt: str, text: Optional[str] = None) -> str:
    if text:

        def hook():
            readline.insert_text(text)
            readline.redisplay()

        readline.set_pre_input_hook(hook)

    result = input(prompt)

    if text:
        readline.set_pre_input_hook()

    return result


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
        key=safe_get(["key"], raw),
        issue_type=safe_get(["fields", "issuetype", "name"], raw),
        summary=safe_get(["fields", "summary"], raw),
        status=safe_get(["fields", "status", "name"], raw),
        created=created.split("T")[0] if created else None,
        parent_key=safe_get(["fields", "parent", "key"], raw),
        subtask_keys=[
            cast(str, s["key"])
            for s in safe_get(
                ["fields", "subtasks"],
                raw,
                [],
            )
        ],
        assignee=safe_get(["fields", "assignee", "displayName"], raw),
        raw=raw,
    )


def _parse_issues(body: dict) -> List[Issue]:
    try:
        return sorted(
            [_parse_issue(issue) for issue in body["issues"]],
            key=lambda i: "_".join(
                [i.status + i.issue_type + (i.created or "")],
            ),
        )
    except Exception as e:
        logger.error("could not parse issues", e, body)
        return []


def _print_table(rows: List[IssueFields]) -> None:
    if len(rows) == 0:
        return
    widths = [max(map(len, col)) for col in zip(*rows)]
    for row in rows:
        print("  ".join((val.ljust(width) for val, width in zip(row, widths))))


def _issues_to_rows(issues: List[Issue], indentation=0) -> List[IssueFields]:
    return [_issue_fields(issue, indentation) for issue in issues]


def _issue_fields(issue: Issue, indentation=0) -> IssueFields:
    indent = "  " * indentation
    return [
        indent + issue.key,
        issue.issue_type,
        issue.status,
        issue.summary,
        issue.created or "",
        issue.assignee or "",
        _issue_browse_url(issue),
    ]


def _issue_single_line(issue: Issue):
    print(f"{issue.key} {issue.summary} -> {issue.status} {issue.assignee}")


def _slugify(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^a-z0-9-]", "", s)
    return s


def _issue(issue_key: str) -> Issue:
    issue_url = _api_url(f"issue/{issue_key}")
    body = _api_get(issue_url)
    return _parse_issue(body)


def _epic_issues(issue: Issue) -> List[Issue]:
    return _search(f'"Epic Link" = "{issue.key}" AND statusCategory != "Done"')


def _children(issue: Issue) -> List[Issue]:
    if issue.is_epic():
        child_issues = _epic_issues(issue)
    else:
        child_issues = [_issue(k) for k in (issue.subtask_keys or [])]
    return [i for i in child_issues if i and i.status not in ["Done", "Invalid"]]


def _issue_key_or_select(
    issue_key: Optional[str],
    cmd: List[str],
) -> Optional[str]:
    if issue_key:
        return issue_key
    else:
        issue = _select_issue(cmd)
        issue_key = issue.key if issue else None
    return issue_key


def _issue_browse_url(issue: Union[str, Issue]) -> str:
    issue_key = issue if isinstance(issue, str) else issue.key
    return f"{_load_config().base_url}/browse/{issue_key}"


def _search(query: str) -> List[Issue]:
    search_url = _api_url("search")
    params = {"jql": query}
    body = _api_get(search_url, params=params)
    return _parse_issues(body)


def _todo() -> List[Issue]:
    cfg = _load_config()
    return _search(
        f'project = {cfg.project_key} AND status = "{cfg.todo_status}"',
    )


def _mine() -> List[Issue]:
    project_key = _load_config().project_key
    return _search(
        f"project = {project_key} AND assignee = currentUser() AND statusCategory != Done"
    )


def _issue_from_branch() -> Tuple[str, str]:
    cfg = _load_config()
    branch_name = (
        subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            check=True,
            capture_output=True,
        )
        .stdout.decode()
        .strip()
    )
    issue_type, rest = branch_name.split("/", maxsplit=1)
    issue_key_match = re.search(rf"({cfg.project_key}-\d+)", rest)
    issue_key = issue_key_match.group(1) if issue_key_match else None
    if not issue_key:
        logging.error("could not find issue key in branch name")
        raise Exception("could not find issue key in branch name")
    return (issue_type, issue_key)


def _branch_name(
    issue: Union[str, Issue],
    prefix: Optional[str] = None,
) -> str:
    if isinstance(issue, str):
        issue = _issue(issue)
    prefix = prefix or ("fix" if issue.is_bug() else "feat")
    branch_name = f"{prefix}/{issue.key}-{_slugify(issue.summary)}"
    branch_name = branch_name[:80]
    return branch_name


def _create_branch(branch_name: str, force=False):
    final_branch_name = (
        branch_name if force else _input("Create branch: ", branch_name).strip()
    )
    if final_branch_name:
        subprocess.run(
            ["git", "checkout", "-b", final_branch_name],
            check=True,
        )


def _select_issue(list_issues_cmd: List[str]) -> Optional[Issue]:
    list_issues = subprocess.Popen(
        list_issues_cmd,
        stdout=subprocess.PIPE,
        text=True,
    )
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
        ["fzf"],
        stdin=list_transitions.stdout,
        stdout=subprocess.PIPE,
        text=True,
    )
    output, error = fzf.communicate()
    if error:
        raise Exception(error)
    transition_id, transition_name = (
        output.strip().split(" ", maxsplit=1) if output else (None, None)
    )
    if transition_id and transition_name:
        return Transition(transition_id, transition_name)


def _collect_text_input(initial: Optional[str] = None) -> str:
    with tempfile.NamedTemporaryFile(suffix=".tmp") as temp_file:
        temp_file_path = temp_file.name
        if initial:
            with open(temp_file_path, "w") as f:
                f.write(initial)
        subprocess.run(
            [os.environ.get("VISUAL", "vi"), temp_file_path],
            check=True,
        )
        with open(temp_file_path, "r") as f:
            return f.read().strip()


def _assign(issue_key: str, user: Optional[str]):
    issue_url = _api_url(f"issue/{issue_key}/assignee")
    body = {"name": user}
    requests.put(issue_url, json=body, headers=_headers()).raise_for_status()


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
    issue_type: str,
    summary: str,
    description: Optional[str] = None,
    fill: bool = False,
    parent_key: Optional[str] = None,
) -> Issue:
    cfg = _load_config()

    if not description and not fill:
        description = _collect_text_input()

    body = {
        "fields": {
            "project": {"key": cfg.project_key},
            "summary": summary,
            "issuetype": {"name": issue_type},
        }
    }
    if description:
        body["fields"]["description"] = description
    if parent_key:
        body["fields"]["parent"] = {"key": parent_key}
        body["fields"]["issuetype"] = {"name": "Sub-task"}
    response = requests.post(_api_url("issue"), json=body, headers=_headers())
    if response.status_code != 201:
        raise Exception(f"Error creating issue: {response.text}")
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
        _create_branch(_branch_name(issue_key, prefix), force=args.force)


def move(args: argparse.Namespace):
    issue_key = _issue_key_or_select(args.issue, ["jira", "mine"])
    status = safe_get("status", args)
    if issue_key and status:
        status = safe_get(
            "0",
            ([s for s in _transitions(issue_key) if s.name.lower() == status.lower()]),
            None,
        )
    if issue_key and not status:
        status = _select_transition(issue_key)
    if issue_key and status:
        _move(issue_key, status.id)
        _issue_single_line(_issue(issue_key))


def create(args: argparse.Namespace):
    issue = _create_issue(
        args.type, args.summary, args.description, args.fill, args.parent
    )
    print(f"Created {issue.key} {issue.summary}")
    if args.work:
        work(argparse.Namespace(issue=issue.key))
    if args.branch:
        branch(argparse.Namespace(issue=issue.key, force=False))


def work(args: argparse.Namespace):
    issue_key = _issue_key_or_select(args.issue, ["jira", "todo"])
    if not issue_key:
        return
    cfg = _load_config()
    _move(issue_key, cfg.in_progress_id)
    _assign(issue_key, cfg.user)
    issue = _issue(issue_key)
    _issue_single_line(issue)


def drop(args: argparse.Namespace):
    issue_key = _issue_key_or_select(args.issue, ["jira", "mine"])
    if not issue_key:
        return
    cfg = _load_config()
    _move(issue_key, cfg.todo_id)
    _assign(issue_key, None)
    issue = _issue(issue_key)
    _issue_single_line(issue)


def done(args: argparse.Namespace):
    issue_key = _issue_key_or_select(args.issue, ["jira", "mine"])
    if not issue_key:
        return
    cfg = _load_config()
    _move(issue_key, cfg.done_id)
    issue = _issue(issue_key)
    _issue_single_line(issue)


def dump_issue(args: argparse.Namespace):
    cmd = safe_get("list_cmd", args) or "mine"
    issue_key = _issue_key_or_select(args.issue, ["jira", cmd])
    if not issue_key:
        return
    issue = _issue(issue_key)
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


def _prefixed_issue(issue_key: str, prefix: str, component: Optional[str]) -> str:
    full_prefix = "".join(
        [
            p
            for p in [
                prefix,
                f"({component})" if component else None,
                ":",
            ]
            if p
        ]
    )

    return f"{full_prefix} {issue_key}"


def current(args: argparse.Namespace):
    issue_type, issue_key = _issue_from_branch()
    if args.full:
        _print_table([_issue_fields(_issue(issue_key))])
        return

    prefix = safe_get("prefix", args)
    component = safe_get("component", args)

    print(_prefixed_issue(issue_key, prefix or issue_type, component))


def create_pr(_):
    issue_type, issue_key = _issue_from_branch()
    issue = _issue(issue_key)
    title = (
        _prefixed_issue(issue_key, issue_type, safe_get("component", args))
        + " "
        + issue.summary
    )

    parts = [
        s
        for s in _collect_text_input(title)
        .strip()
        .split(
            "\n",
            maxsplit=1,
        )
        if s
    ]

    git_cmd: List[str] = [
        "gh",
        "pr",
        "create",
        "--fill",
    ]
    cfg = _load_config()
    for l in cfg.default_pr_labels:
        git_cmd.append("--label")
        git_cmd.append(l)

    if not parts:
        return
    if len(parts) == 1:
        git_cmd.append("--title")
        git_cmd.append(parts[0])
    else:
        git_cmd.append("--title")
        git_cmd.append(parts[0])
        git_cmd.append("--body")
        git_cmd.append(parts[1])

    subprocess.run(git_cmd, check=True)


def _tree(issue: Issue) -> IssueNode:
    return IssueNode(issue, [_tree(c) for c in _children(issue)])


def _indented_lines(node: IssueNode, level=0) -> List[IssueFields]:
    lines = [_issue_fields(node.issue, level)]
    for child in node.children:
        lines.extend(_indented_lines(child, level + 1))
    return lines


def tree(args: argparse.Namespace):
    cmd = safe_get("list_cmd", args, "mine")
    issue_key = _issue_key_or_select(args.issue, ["jira", cmd])
    if not issue_key:
        return
    issue = _issue(issue_key)
    tree = _tree(issue)
    _print_table(_indented_lines(tree))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="jira", description="jira cli")

    subparsers = parser.add_subparsers(dest="command")

    # branch
    branch_parser = subparsers.add_parser(
        "branch", help="create a branch from an issue"
    )
    branch_parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="force",
    )
    branch_parser.add_argument(
        "-i",
        "--issue",
        required=False,
        help="issue key",
    )
    branch_parser.add_argument(
        "-p",
        "--prefix",
        required=False,
        help="branch prefix",
    )
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
        "-t",
        "--type",
        type=str,
        required=False,
        help="issue type",
        default="Task",
    )
    create_parser.add_argument(
        "-b",
        "--branch",
        action="store_true",
        help="create a branch for the issue",
    )
    create_parser.add_argument(
        "-f", "--fill", action="store_true", help="fill or skip issue fields"
    )
    create_parser.add_argument(
        "-d",
        "--description",
        type=str,
        help="issue description.",
    )
    create_parser.add_argument(
        "-w",
        "--work",
        action="store_true",
        help="assign issue and move to in progress",
    )
    create_parser.set_defaults(func=create)

    # dump_issue
    dump_issue_parser = subparsers.add_parser(
        "dump_issue", help="dump the raw json of an issue"
    )
    dump_issue_parser.add_argument("-i", "--issue", required=False, help="issue key")
    dump_issue_parser.add_argument("list_cmd", nargs="?", help="list command")
    dump_issue_parser.set_defaults(func=dump_issue)

    # mine
    mine_parser = subparsers.add_parser("mine", help="list my issues")
    mine_parser.set_defaults(func=mine)

    # open
    open_parser = subparsers.add_parser("open", help="open issue in browser")
    open_parser.add_argument("-i", "--issue", required=False, help="issue key")
    open_parser.add_argument(
        "-l",
        "--list-cmd",
        required=False,
        help="list command",
    )
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
    work_parser = subparsers.add_parser(
        "work",
        help="start working on an issue",
    )
    work_parser.add_argument("issue", nargs="?", help="issue key")
    work_parser.set_defaults(func=work)

    # drop
    drop_parser = subparsers.add_parser(
        "drop",
        help="drop an issue",
    )
    drop_parser.add_argument("issue", nargs="?", help="issue key")
    drop_parser.set_defaults(func=drop)

    # url
    url_parser = subparsers.add_parser("url", help="get issue url")
    url_parser.add_argument(
        "-l", "--list-cmd", type=str, required=False, help="list command"
    )
    url_parser.add_argument("issue", nargs="?", help="issue key")
    url_parser.set_defaults(func=url)

    # done
    done_parser = subparsers.add_parser(
        "done",
        help="finish working on an issue",
    )
    done_parser.add_argument("issue", nargs="?", help="issue key")
    done_parser.set_defaults(func=done)

    # current
    current_parser = subparsers.add_parser(
        "current", help="print info about the issue of the current branch"
    )
    current_parser.add_argument(
        "-f", "--full", action="store_true", help="print full issue data"
    )
    current_parser.add_argument("-c", "--component", help="component")
    current_parser.add_argument("prefix", nargs="?", help="prefix")
    current_parser.set_defaults(func=current)

    # pr
    pr_parser = subparsers.add_parser(
        "pr", help="create a PR for the current branch issue"
    )
    pr_parser.add_argument("-c", "--component", help="component")
    pr_parser.set_defaults(func=create_pr)

    # tree
    tree_parser = subparsers.add_parser("tree", help="dumps issue tree")
    tree_parser.add_argument(
        "-l", "--list-cmd", type=str, required=False, help="list command"
    )
    tree_parser.add_argument("issue", nargs="?", help="issue key")
    tree_parser.set_defaults(func=tree)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        exit(1)

    # print(args)
    args.func(args)
