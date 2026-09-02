#!/usr/bin/env python3

import argparse
import re
import sys
from datetime import datetime


MONTHS = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}


def parse_timestamp(ts):

    m = re.match(
        r"\w+\s+(\w+)\s+(\d+)\s+(\d+):(\d+):(\d+)\s+(\d+)",
        ts,
    )

    if not m:
        return None

    return datetime(
        int(m.group(6)),
        MONTHS[m.group(1)],
        int(m.group(2)),
        int(m.group(3)),
        int(m.group(4)),
        int(m.group(5)),
    )


def duration(start, end):

    if not start or not end:
        return "unknown"

    delta = int((end - start).total_seconds())

    h = delta // 3600
    m = (delta % 3600) // 60
    s = delta % 60

    return f"{h:02d}:{m:02d}:{s:02d}"


def parse_log(logfile):

    jobs = {}
    job_stats = {}
    errors = []

    start_time = None
    end_time = None

    workflow_status = "UNKNOWN"
    workflow_log = None
    steps_done = None
    steps_total = None

    with open(logfile, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    #
    # PASS 1
    # Extract all rule blocks
    #
    i = 0

    while i < len(lines):

        line = lines[i].rstrip("\n")

        #
        # Timestamp
        #
        m = re.match(r"^\[(.+)\]$", line)

        if m:

            ts = parse_timestamp(m.group(1))

            if ts:
                if start_time is None:
                    start_time = ts

                end_time = ts

        #
        # Job stats
        #
        if line.strip() == "Job stats:":

            i += 1

            while i < len(lines) and not lines[i].strip():
                i += 1

            #
            # skip header
            #
            if i + 1 < len(lines):
                i += 2

            while i < len(lines):

                l = lines[i].rstrip()

                if not l.strip():
                    break

                m = re.match(r"(.+?)\s+(\d+)$", l)

                if m:
                    job_stats[m.group(1).strip()] = int(m.group(2))

                i += 1

            continue

        #
        # Rule block
        #
        m = re.match(r"^\s*rule\s+(\S+):", line)

        if m:

            rule_name = m.group(1)

            block = {
                "rule": rule_name,
                "jobid": None,
                "input": "",
                "output": "",
                "wildcards": "",
                "slurm_jobid": None,
                "status": "UNKNOWN",
            }

            j = i + 1

            while j < len(lines):

                l = lines[j].rstrip("\n")

                #
                # end of block
                #
                if not l.startswith((" ", "\t")):
                    break

                m2 = re.match(r"\s*jobid:\s*(\d+)", l)
                if m2:
                    block["jobid"] = int(m2.group(1))

                m2 = re.match(r"\s*input:\s*(.*)", l)
                if m2:
                    block["input"] = m2.group(1).strip()

                m2 = re.match(r"\s*output:\s*(.*)", l)
                if m2:
                    block["output"] = m2.group(1).strip()

                m2 = re.match(r"\s*wildcards:\s*(.*)", l)
                if m2:
                    block["wildcards"] = m2.group(1).strip()

                j += 1

            if block["jobid"] is not None:
                jobs[block["jobid"]] = block

        i += 1

    #
    # PASS 2
    # SLURM mapping + status
    #
    in_group = False
    pending_group_jobids = []

    for line in lines:

        #
        # Group job header - start tracking inner jobids
        #
        if re.search(r"Group job .+ \(jobs in lexicogr\. order\):", line):
            in_group = True
            pending_group_jobids = []

        #
        # Collect inner jobids when inside a group block
        #
        if in_group:
            m = re.match(r"\s+jobid:\s*(\d+)", line)
            if m:
                jid = int(m.group(1))
                if jid in jobs:
                    pending_group_jobids.append(jid)

        #
        # Slurm submission - integer jobid (regular jobs)
        #
        m = re.search(
            r"Job\s+(\d+)\s+has been submitted with SLURM jobid\s+(\d+)",
            line,
        )

        if m:

            smk_jobid = int(m.group(1))
            slurm_jobid = int(m.group(2))

            if smk_jobid in jobs:
                jobs[smk_jobid]["slurm_jobid"] = slurm_jobid

            in_group = False
            pending_group_jobids = []

        #
        # Slurm submission - UUID jobid (group jobs)
        #
        m = re.search(
            r"Job\s+[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}"
            r"\s+has been submitted with SLURM jobid\s+(\d+)",
            line,
            re.IGNORECASE,
        )

        if m:

            slurm_jobid = int(m.group(1))

            for jid in pending_group_jobids:
                if jid in jobs:
                    jobs[jid]["slurm_jobid"] = slurm_jobid

            in_group = False
            pending_group_jobids = []

        #
        # Finished job
        #
        m = re.search(r"Finished jobid:\s+(\d+)", line)

        if m:

            jobid = int(m.group(1))

            if jobid in jobs:
                jobs[jobid]["status"] = "OK"

        #
        # Workflow completion
        #
        m = re.search(
            r"(\d+)\s+of\s+(\d+)\s+steps\s+\((\d+)%\)\s+done",
            line,
        )

        if m:
            workflow_status = "SUCCESS"
            steps_done = int(m.group(1))
            steps_total = int(m.group(2))

        #
        # Complete log
        #
        m = re.match(r"Complete log\(s\):\s+(.*)", line)

        if m:
            workflow_log = m.group(1).strip()

    #
    # Traceback extraction
    #
    i = 0

    while i < len(lines):

        if lines[i].startswith("Traceback (most recent call last):"):

            tb = [lines[i].rstrip()]

            i += 1

            while i < len(lines):

                tb.append(lines[i].rstrip())

                if re.match(r"^\w+Error:", lines[i]):
                    break

                if re.match(r"^\w+Exception:", lines[i]):
                    break

                i += 1

            errors.append("\n".join(tb))

        i += 1

    #
    # Remaining jobs
    #
    for job in jobs.values():

        if job["status"] == "UNKNOWN":

            if job["slurm_jobid"]:
                job["status"] = "SUBMITTED"
            else:
                job["status"] = "LOCAL/UNKNOWN"

    if errors:
        workflow_status = "FAILED"

    return {
        "jobs": jobs,
        "job_stats": job_stats,
        "errors": errors,
        "start_time": start_time,
        "end_time": end_time,
        "workflow_status": workflow_status,
        "workflow_log": workflow_log,
        "steps_done": steps_done,
        "steps_total": steps_total,
    }


def print_report(data, long_output=False, logfile=None):

    print("=" * 80)
    print("Snakemake Execution Report")
    print("=" * 80)
    print()

    print(f"Log file   : {logfile}")
    print(f"Start time : {data['start_time']}")
    print(f"End time   : {data['end_time']}")
    print(
        f"Duration   : "
        f"{duration(data['start_time'], data['end_time'])}"
    )
    print()

    print(f"Status     : {data['workflow_status']}")

    if data["steps_done"] is not None:
        print(
            f"Steps      : "
            f"{data['steps_done']} / {data['steps_total']}"
        )

    print()

    print("Job statistics")
    print("-" * 80)

    for rule, count in sorted(data["job_stats"].items()):
        print(f"{rule:<30} {count}")

    print()

    print("Executed jobs")
    print("-" * 80)

    header = (
        f"{'SMK_JOB':>8} "
        f"{'RULE':<25} "
        f"{'SLURM_JOB':>12} "
        f"{'STATUS':<15}"
    )

    print(header)
    print("-" * len(header))

    for jobid in sorted(data["jobs"]):

        j = data["jobs"][jobid]

        slurm = (
            str(j["slurm_jobid"])
            if j["slurm_jobid"]
            else "local"
        )

        print(
            f"{jobid:8d} "
            f"{j['rule']:<25} "
            f"{slurm:>12} "
            f"{j['status']:<15}"
        )

    print()

    if long_output:

        print("Detailed job information")
        print("-" * 80)

        for jobid in sorted(data["jobs"]):

            j = data["jobs"][jobid]

            print()
            print(f"JobID      : {jobid}")
            print(f"Rule       : {j['rule']}")
            print(f"SLURM      : {j['slurm_jobid']}")
            print(f"Status     : {j['status']}")

            if j["wildcards"]:
                print(f"Wildcards  : {j['wildcards']}")

            if j["input"]:
                print(f"Input      : {j['input']}")

            if j["output"]:
                print(f"Output     : {j['output']}")

    if data["errors"]:

        print()
        print("Errors")
        print("-" * 80)

        for err in data["errors"]:

            print()
            print(err)

    if data["workflow_log"]:

        print()
        print("Workflow log")
        print("-" * 80)
        print(data["workflow_log"])


def main():

    parser = argparse.ArgumentParser(
        description="Parse and report a Snakemake log file."
    )
    parser.add_argument("logfile", help="Path to the Snakemake log file")
    parser.add_argument(
        "-l",
        action="store_true",
        dest="long_output",
        help="Enable long (detailed) output",
    )

    args = parser.parse_args()
    long_output = args.long_output

    data = parse_log(args.logfile)

    print_report(data, long_output=long_output, logfile=args.logfile)


if __name__ == "__main__":
    main()
