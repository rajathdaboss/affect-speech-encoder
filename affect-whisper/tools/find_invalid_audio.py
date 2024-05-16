from dataset import WTCAudioDataset

print('Loading WTC Audio Dataset...')
dataset = WTCAudioDataset(emb_space='affect')
print('\tDone.\n')

for i in range(len(dataset)):
    print(f'Attempting [{i}]')
    try:
        log_mel, emb = dataset[i]
        print('\tSuccess.')
    except:
        with open('bad_audios.txt', 'a') as f:
            f.write(f'{i}\n')
        print('\nFailed.')