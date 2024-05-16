import os, argparse
import numpy as np
import pandas as pd
from mysql import connector


def run(cursor, emb_space='affect'):
    all_segments = []
    print('Preparing Whisper Audio Segments...')
    for batch in range(1, 4):
        with open(f'assets/batch{batch}.txt', 'r') as f:
            for line in f.readlines():
                filename = line.strip()
                segments = get_segments(cursor, batch, filename)
                if segments:
                    all_segments.extend(segments)
    print('\tDone.\n')

    df = pd.read_csv(f'/data/rrao/affects/clinic_audio_segment_{emb_space}.csv')
    
    print('Extracting Ground Truth Embeddings...')
    for idx, (filename, batch, segment, start, end) in enumerate(all_segments):
        print(f'[{idx}] ({segment}) Retrieving Embeddings...\t', end='')
        # Extract Ground Truth Embeddings
        if emb_space == 'affect':
            query = f"SELECT group_norm FROM feat$cat_dd_affInt_w$msgs_whisper_large_dia2_p{batch}$message_id$1gra WHERE group_id = '{segment}';"
            emb_size = 2
        elif emb_space == 'roberta':
            query = f"SELECT group_norm FROM feat$roberta_la_meL23con$msgs_whisper_large_dia2_p{batch}$message_id WHERE group_id = '{segment}';"
            emb_size = 1024
        cursor.execute(query)

        data = [affect[0] for affect in cursor.fetchall()]
        if len(data) == emb_size:
            data = [segment, batch, filename, start, end] + data
            row = pd.DataFrame([data], columns=df.columns)
            row.to_csv(f'/data/rrao/affects/clinic_audio_segment_{emb_space}.csv', mode='a', header=False, index=False)
            print('Success.')
        else:
            print('Failed.')


def get_segments(cursor, batch, filename):
    query = f"SELECT message_id, startTime, endTime FROM msgs_whisper_large_diarized2_p{batch} WHERE filename = '{filename[:-4] + '.json'}';"
    cursor.execute(query)
    segments = cursor.fetchall()
    return [(filename, batch, segment[0], segment[1], segment[2]) for segment in segments]


# Verifies user's credentials and returns a MySQL Connection object
def get_sql_credentials(filename):
    usr = ''
    pwd = ''
    with open(filename, 'r') as file:
        for line in file.readlines():
            if line.startswith('user'):
                usr = line[5:].strip()
            if line.startswith('password'):
                pwd = line[9:].strip()

    connection = connector.connect(
        host='localhost',
        user=usr,
        password=pwd,
        database='wtc_clinic'
    )
    return connection, connection.cursor()


if __name__ == "__main__":
    connection, cursor = get_sql_credentials('/users2/rrao/.my.cnf')

    run(cursor, emb_space='roberta')

    connection.close()
    cursor.close()