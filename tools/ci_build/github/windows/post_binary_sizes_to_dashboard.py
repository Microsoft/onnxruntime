#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


import argparse
import mysql.connector
import sys
import os


def parse_arguments():
    parser = argparse.ArgumentParser(description="ONNXRuntime binary size uploader for dashboard")
    parser.add_argument("--commit_hash", help="Full Git commit hash")
    parser.add_argument("--build_project", default='Lotus', choices=['Lotus', 'onnxruntime'],
                        help="Lotus or onnxruntime build project, to construct the build URL")
    parser.add_argument("--build_id", help="Build Id")
    parser.add_argument("--size_data_file", help="Path to file that contains the binary size data")
    parser.add_argument("--ignore_db_error", action='store_true',
                        help="Ignore database errors while executing this script")
    parser.add_argument("--database_server", help="The hostname or ip address of the database server")
    return parser.parse_args()

# Assumes size_data_file is a csv file with a header line, containing binary sizes and other attributes
# CSV fields are:
#    os,arch,build_config,size
# No empty line or space between fields expected


def get_binary_sizes(size_data_file):
    binary_size = []
    with open(size_data_file, 'r') as f:
        line = f.readline()
        headers = line.strip().split(',')
        while line:
            line = f.readline()
            if not line:
                break
            linedata = line.strip().split(',')
            tablerow = {}
            for i in range(0, len(headers)):
                if headers[i] == 'size':
                    tablerow[headers[i]] = int(linedata[i])
                else:
                    tablerow[headers[i]] = linedata[i]
            binary_size.append(tablerow)
    return binary_size


def write_to_db(binary_size_data, args):
    # connect to database

    cnx = mysql.connector.connect(
        user='ort@onnxruntimedashboard',
        password=os.environ.get('DASHBOARD_MYSQL_ORT_PASSWORD'),
        host=args.database_server,
        database='onnxruntime')

    try:
        cursor = cnx.cursor()

        # insert current records
        for row in binary_size_data:
            insert_query = ('INSERT INTO onnxruntime.binary_size '
                            '(build_time, build_project, build_id, commit_id, os, arch, build_config, size) '
                            'VALUES (Now(), "%s", "%s", "%s", "%s", "%s", "%s", %d) '
                            'ON DUPLICATE KEY UPDATE '
                            'build_time=Now(), build_project="%s", build_id="%s", size=%d;'
                            ) % (
                args.build_project,
                args.build_id,
                args.commit_hash,
                row['os'],
                row['arch'],
                row['build_config'],
                row['size'],

                args.build_project,
                args.build_id,
                row['size']
            )
            cursor.execute(insert_query)

        cnx.commit()

        # # Use below for debugging:
        # cursor.execute('select * from onnxruntime.binary_size')
        # for r in cursor:
        #     print(r)

        cursor.close()
        cnx.close()
    except BaseException as e:
        cnx.close()
        raise e


if __name__ == "__main__":
    args = parse_arguments()
    binary_size_data = get_binary_sizes(args.size_data_file)
    try:
        write_to_db(binary_size_data, args)
    except BaseException as e:
        print(str(e))
        # If there is DB connection error, and we choose '--ignore_db_error'
        # we can let the script exit clean in order not to fail the pipeline
        if not args.ignore_db_error:
            sys.exit(1)
