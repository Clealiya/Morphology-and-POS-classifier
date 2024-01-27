# split setences in order to have sequences with the same lenght (add padding)

from typing import List, Callable
from icecream import ic


Word_Info = List[str]       # example: ['6', 'flights', 'flight', 'NOUN', '_', 'Number=Plur', '1', 'obj', '_', '_']
Sentence = List[Word_Info]  # list of word info
Sequence = List[Word_Info]  # list of word info with the same length


def dummy_sequences(sentence: Sentence, k: int, pad: str) -> List[Sequence]:
    """ converte one Sentence in a list of Sequences of length k
    idee: add padding at the end of Sentence (with pad caracter) in order to have 
            a sentences length = k * q  s.t.  q in N. 
    return a list of q Sequences
    """
    n = len(sentence)
    assert n != 0, f"Error, len(sentence) == 0"

    num_pad_2_add =  k - n % k
    q = (n + num_pad_2_add) // k

    # padding = [pad, pad, pad] if word is a list, just [pad] otherwise
    padding = [len(sentence[0]) * [pad]] if type(sentence[0]) == list else [pad]
    # add number pad to add * padding at the end of setence
    sentence = sentence + num_pad_2_add * padding

    sequences = []
    for i in range(q):
        sequences.append(sentence[i * k : (i + 1) * k])
    
    return sequences


def create_sequences(sentences: List[Sentence],
                     sequence_function: Callable[[Sentence, int, str], List[Sentence]],
                     k: int,
                     pad: str,
                     ) -> List[Sequence]:
    """ run a sequence function like dummy_sequences on a list of Sentences """
    sequences = []
    for sentence in sentences:
        sequences += sequence_function(sentence, k, pad)
    return sequences


def find_sequence_function(name: str) -> Callable[[Sentence, int, str], List[Sentence]]:
    if name == 'dummy':
        return dummy_sequences
    else:
        raise NotImplementedError(f"Error, the function '{name}' to split senteces and get sequences was not implemented")


if __name__ == '__main__':
    sentence = [['1', 'i', 'Case=Nom|Number=Sing|Person=1|PronType=Prs'],
                ['2', 'need', 'Mood=Ind|Tense=Pres|VerbForm=Fin'],
                ['3', 'a', 'PronType=Art'],
                ['4', 'list', 'Number=Sing'],
                ['5', 'of', '_'],
                ['6', 'late', 'Degree=Pos'],
                ['7', 'afternoon', 'Number=Sing'],
                ['8', 'flights', 'Number=Plur'],
                ['9', 'from', '_'],
                ['10', 'st.', 'Number=Sing'],
                ['11', 'louis', 'Number=Sing'],
                ['12', 'to', '_'],
                ['13', 'chicago', 'Number=Sing']]
    #print(sentence)
    sequences = dummy_sequences(sentence, k=4, pad='<PAD>')
    print(sequences)
