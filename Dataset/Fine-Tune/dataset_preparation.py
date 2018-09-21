from re import sub
from os import listdir
from os import path
from io import open

from spacy.lang.en import English

tokenizer = English()
tokenizer.add_pipe(tokenizer.create_pipe('sentencizer'))

contents = listdir('.')

for file in contents:
    if path.isdir(file):
        sentences = []
        speeches = listdir(file)

        for speech in speeches:
            with open(path.join(file, speech), "r") as f:
                raw_text = f.read()

                # Remove metadata at top of file
                raw_text = sub('<.*?>', '', raw_text)

                doc = tokenizer(raw_text)
                sentences += [sent.string.strip() for sent in doc.sents]

        with open(file + '.txt', "w") as f:
            for sentence in sentences:
                doc = tokenizer(sentence)

                words = ' '.join([token.text.lower() for token in doc]) + '\n'
                f.write(words)
