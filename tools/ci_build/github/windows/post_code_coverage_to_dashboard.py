#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


# command line arguments
# --report_url=<string>
# --report_file=<string, local file path, TXT/JSON file>
# --commit_hash=<string, full git commit hash>

import argparse
import mysql.connector
import json
import sys
import os


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="ONNXRuntime test coverge report uploader for dashboard")
    parser.add_argument("--report_url", type=str, help="URL to the LLVM json report")
    parser.add_argument(
        "--report_file", type=str, help="Path to the local JSON/TXT report", required=True)
    parser.add_argument("--commit_hash", type=str, help="Full Git commit hash", required=True)
    parser.add_argument("--branch", type=str, help="Source code branch")
    parser.add_argument("--os", type=str, help="Build configuration:os")
    parser.add_argument("--arch", type=str, help="Build configuration:arch")
    parser.add_argument("--build_config", type=str, help="Build configuration: build variants")
    parser.add_argument("--database_server", help="The hostname or ip address of the database server")
    return parser.parse_args()


def parse_txt_report(report_file):
    data = {}
    with open(report_file, 'r') as report:
        for line in reversed(report.readlines()):
            if 'TOTAL' in line:
                fields = line.strip().split()
                data['lines_valid'] = int(fields[1])
                data['lines_covered'] = int(fields[2])
                data['coverage'] = float(fields[3].strip('%'))/100
                break
    return data


def parse_json_report(report_file):
    result = {}
    with open(report_file) as json_file:
        data = json.load(json_file)

    linestat = data['data'][0]['totals']['lines']
    result['coverage'] = float(linestat['percent']/100.0)
    result['lines_covered'] = int(linestat['covered'])
    result['lines_valid'] = int(linestat['count'])
    return result


def write_to_db(coverage_data, args):
    # connect to database

    cnx = mysql.connector.connect(
        user='ort@onnxruntimedashboard',
        password=os.environ.get('DASHBOARD_MYSQL_ORT_PASSWORD'),
        host=args.database_server,
        database='onnxruntime')

    try:
        cursor = cnx.cursor()

        # delete old records
        delete_query = ('DELETE FROM onnxruntime.test_coverage '
                        'WHERE UploadTime < DATE_SUB(Now(), INTERVAL 30 DAY);'
                        )

        cursor.execute(delete_query)

        # insert current record
        insert_query = ('INSERT INTO onnxruntime.test_coverage '
                        '''(UploadTime, CommitId, Coverage, LinesCovered, TotalLines, OS,
                          Arch, BuildConfig, ReportURL, Branch) '''
                        'VALUES (Now(), "%s", %f, %d, %d, "%s", "%s", "%s", "%s", "%s") '
                        'ON DUPLICATE KEY UPDATE '
                        '''UploadTime=Now(), Coverage=%f, LinesCovered=%d, TotalLines=%d,
                          OS="%s", Arch="%s", BuildConfig="%s", ReportURL="%s", Branch="%s"; '''
                        ) % (args.commit_hash,
                             coverage_data['coverage'],
                             coverage_data['lines_covered'],
                             coverage_data['lines_valid'],
                             args.os.lower(),
                             args.arch.lower(),
                             args.build_config.lower(),
                             args.report_url.lower(),
                             args.branch.lower(),
                             coverage_data['coverage'],
                             coverage_data['lines_covered'],
                             coverage_data['lines_valid'],
                             args.os.lower(),
                             args.arch.lower(),
                             args.build_config.lower(),
                             args.report_url.lower(),
                             args.branch.lower()
                             )
        cursor.execute(insert_query)
        cnx.commit()

        # # Use below for debugging:
        # cursor.execute('select * from onnxruntime.test_coverage')
        # for r in cursor:
        #     print(r)

        cursor.close()
        cnx.close()
    except BaseException as e:
        cnx.close()
        raise e


if __name__ == "__main__":
    try:
        args = parse_arguments()
        if args.report_file.endswith(".json"):
            coverage_data = parse_json_report(args.report_file)
        elif args.report_file.endswith(".txt"):
            coverage_data = parse_txt_report(args.report_file)
        else:
            raise ValueError("Only report extensions txt or json are accepted")

        write_to_db(coverage_data, args)
    except BaseException as e:
        print(str(e))
        sys.exit(1)
