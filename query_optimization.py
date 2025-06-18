import os
import logging
import re
import json
import psycopg2

# TODO define your POSTGRES_DATA_DIR
POSTGRES_DATA_DIR = '/root/pg_data/'
# TODO define output result dir
ROOT_DIR = os.path.dirname(__file__)
# TODO define the pg instance connector.
db_name = 'imdb'
connection = psycopg2.connect(database=db_name, user='USER', password='PASSWORD', host="127.0.0.1", port=5432)

cursor = connection.cursor()
print("connected to the database")


def exec_only(cursor, statement):
    cursor.execute(statement)


def exec_fetch(cursor, statement, one=True):
    cursor.execute(statement)
    if one:
        return cursor.fetchone()
    return cursor.fetchall()


def drop_indexes():
    print("Dropping indexes")
    stmt = "select indexname from pg_indexes where schemaname='public'"
    indexes = exec_fetch(cursor, stmt, one=False)
    for index in indexes:
        index_name = index[0]
        if index_name.endswith('_pkey'):
            continue
        drop_stmt = "drop index {}".format(index_name)
        logging.debug("Dropping index {}".format(index_name))
        exec_only(cursor, drop_stmt)
    connection.commit()


def add_indexes():
    add_stmt = [
        # Indexes on movie_info table
        "CREATE INDEX idx_movie_info_movie_id ON movie_info(movie_id);",
        "CREATE INDEX idx_movie_info_info_type_id ON movie_info(info_type_id);",

        # Indexes on cast_info table
        "CREATE INDEX idx_cast_info_movie_id ON cast_info(movie_id);",
        "CREATE INDEX idx_cast_info_person_id ON cast_info(person_id);",
        "CREATE INDEX idx_cast_info_role_id ON cast_info(role_id);",
        "CREATE INDEX idx_cast_info_person_role_id ON cast_info(person_role_id);",

        # Indexes on movie_keyword table
        "CREATE INDEX idx_movie_keyword_movie_id ON movie_keyword(movie_id);",
        "CREATE INDEX idx_movie_keyword_keyword_id ON movie_keyword(keyword_id);",

        # Indexes on movie_companies table
        "CREATE INDEX idx_movie_companies_movie_id ON movie_companies(movie_id);",
        "CREATE INDEX idx_movie_companies_company_type_id ON movie_companies(company_type_id);",
        "CREATE INDEX idx_movie_companies_company_id ON movie_companies(company_id);",

        # Indexes on movie_info_idx table
        "CREATE INDEX idx_movie_info_idx_movie_id ON movie_info_idx(movie_id);",
        "CREATE INDEX idx_movie_info_idx_info_type_id ON movie_info_idx(info_type_id);",

        # Indexes on aka_title table
        "CREATE INDEX idx_aka_title_movie_id ON aka_title(movie_id);",
        "CREATE INDEX idx_aka_title_kind_id ON aka_title(kind_id);",

        # Indexes on complete_cast table
        "CREATE INDEX idx_complete_cast_movie_id ON complete_cast(movie_id);",
        "CREATE INDEX idx_complete_cast_status_id ON complete_cast(status_id);",
        "CREATE INDEX idx_complete_cast_subject_id ON complete_cast(subject_id);",

        # Indexes on person_info table
        "CREATE INDEX idx_person_info_person_id ON person_info(person_id);",
        "CREATE INDEX idx_person_info_info_type_id ON person_info(info_type_id);",

        # Indexes on aka_name table
        "CREATE INDEX idx_aka_name_person_id ON aka_name(person_id);",

        # Indexes on title table
        "CREATE INDEX idx_title_kind_id ON title(kind_id);"
    ]

    for stmt in add_stmt:
        print("{}".format(stmt))
        exec_only(cursor, stmt)
    connection.commit()


# add_indexes is invoked only once.
# add_indexes()


methods = [
    'grasp',
    'mscn',
    'pg',
    'gt',
]

workloads = [
    '00',
    '01',
    '02',
    '03',
]

timeout = 5 * 60 * 1000

if ROOT_DIR == '':
    ROOT_DIR = './'
os.makedirs(f"{ROOT_DIR}/qo_results", exist_ok=True)
os.makedirs(f"{ROOT_DIR}/qo_results_plans", exist_ok=True)

for workload in workloads:
    current_methods = methods

    for method in current_methods:
        benchname = "imdb_mix_{}".format(workload)

        res_file = open(f"{ROOT_DIR}/qo_results/{benchname}_{method}.txt", 'w')

        # the pre-calculated CardEst file. When running query optimization, 
        # the card value used by PG optimizer will be replaced by the card value in these files
        stb1_file = f"stb1_{benchname}_{method}.txt"
        joins1_file = f"joins1_{benchname}_{method}.txt"

        range_file = f"range_{benchname}.json"

        with open(POSTGRES_DATA_DIR + range_file) as f:
            range_dict = json.load(f)

        time_sum = 0
        for q_name in range_dict:
            if q_name != '5a_5a107.sql':
                continue
            query_file = f"{ROOT_DIR}/../custom_workloads/imdb_mix_{workload}/{q_name}"
            q_file = open(query_file, 'r')
            query = q_file.read()
            statement = f"explain (analyze, buffers, format json) {query}"

            print(method + ": " + workload + "/" + q_name)

            # for PG raw, we disable the plugin and use the original PG optimizer.
            if method == 'pg':
                cursor.execute('SET ml_cardest_enabled=false')
                cursor.execute('SET ml_joinest_enabled=false')
                cursor.execute(statement)
                time = 0
                for i in range(10):
                    plan = cursor.fetchone()[0][0]["Plan"]
                    time = plan["Actual Total Time"]
                    os.makedirs(f"{ROOT_DIR}/qo_results_plans/{benchname}", exist_ok=True)
                    with open(f"{ROOT_DIR}/qo_results_plans/{benchname}/{q_name}_{method}_{i}.json", "w") as f:
                        json.dump(plan, f)
                    print(time)
                time_sum += time
                res_file.write("{}\n".format(time))
                res_file.flush()
                continue

            std_st, std_end = range_dict[q_name]['stb_ranges']
            joins_st, joins_end = range_dict[q_name]['joins_ranges']

            f = open(POSTGRES_DATA_DIR + stb1_file, 'r')
            f_out = open(POSTGRES_DATA_DIR + "stb1_tmp.txt", 'w')
            lines = f.readlines()

            for line in lines[std_st:std_end]:
                f_out.write(line)

            f.close()
            f_out.close()

            f = open(POSTGRES_DATA_DIR + joins1_file, 'r')
            f_out = open(POSTGRES_DATA_DIR + "joins1_tmp.txt", 'w')
            lines = f.readlines()

            for line in lines[joins_st:joins_end]:
                f_out.write(line)

            f.close()
            f_out.close()

            cursor.execute('SET print_sub_queries=false')
            cursor.execute('SET print_single_tbl_queries=false')

            cursor.execute('SET ml_cardest_enabled=true')
            cursor.execute('SET ml_joinest_enabled=true')

            cursor.execute('SET query_no=0')
            cursor.execute('SET join_est_no=0')

            cursor.execute("SET ml_cardest_fname='{}.txt'".format('stb1_tmp'))
            cursor.execute("SET ml_joinest_fname='{}.txt'".format('joins1_tmp'))

            set_timeout = f"set statement_timeout={timeout}"
            cursor.execute(set_timeout)

            try:
                cursor.execute(statement)
                plan = cursor.fetchone()[0][0]["Plan"]
                time = plan["Actual Total Time"]
                os.makedirs(f"{ROOT_DIR}/qo_results_plans/{benchname}", exist_ok=True)
                with open(f"{ROOT_DIR}/qo_results_plans/{benchname}/{q_name}_{method}.json", "w") as f:
                    json.dump(plan, f)
            except Exception as e:
                print('timeout')
                connection.rollback()
                time = timeout

            res_file.write("{}\n".format(time))
            res_file.flush()
            time_sum += time
            print(time)

        res_file.write("sum: {}\n".format(time_sum))
        res_file.close()
