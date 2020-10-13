#  transformer_chatbot
#  Copyright (C) 2018 Golovanov, Tselousov
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

import random
from collections import defaultdict

import nltk
from nltk.corpus import wordnet


def augment_replica(seq):
    _exceptions = ['your', 'persona']
    pos2wn = {'NN': wordnet.NOUN,
              'JJ': wordnet.ADJ,
              'VBP': wordnet.VERB,
              'RB': wordnet.ADV}

    synonyms = defaultdict(list)

    tagged_seq = seq.replace('i ', 'I ')
    tagged_seq = nltk.pos_tag(nltk.word_tokenize(tagged_seq))

    for word, pos in tagged_seq:
        if pos not in pos2wn or word in _exceptions:
            continue

        pos = pos2wn[pos]
        synnets = wordnet.synsets(word, pos=pos)

        for synnet in synnets:
            for syn in synnet.lemma_names():
                if syn != word:
                    synonyms[word].append(syn.replace('_', ' '))
            break
    if synonyms:
        for key, values in synonyms.items():
            seq = seq.replace(key, random.choice(list(values)))

    return seq
