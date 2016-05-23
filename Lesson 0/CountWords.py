# Count words in Python
# Problem
# Implement a function count_words() in Python that takes as input a string s and a number n, and returns the n most frequently-occuring words in s. The return value should be a list of tuples - the top n words paired with their respective counts [(<word>, <count>), (<word>, <count>), ...], sorted in descending count order.
# You can assume that all input will be in lowercase and that there will be no punctuations or other characters (only letters and single separating spaces). In case of a tie (equal count), order the tied words alphabetically.
# E.g.:
# print count_words("betty bought a bit of butter but the butter was bitter",3)
# Output:
# [('butter', 2), ('a', 1), ('betty', 1)]

"""Count words."""

def count_words(s, n):
    """Return the n most frequently occuring words in s."""
    
    # TODO: Count the number of occurences of each word in s
    wordlist = s.split()
    wordfreq = []
    for w in wordlist:
    	wordfreq.append(wordlist.count(w))
	zipped = zip(wordlist, wordfreq)
    zipped_duped = list(set(zipped))
    zipped_sorted = sorted(zipped_duped, key = lambda x: (-x[1], x[0]))
    top_n = zipped_sorted[:n]
    import os
    try:
        user_paths = os.environ['PYTHONPATH'].split(os.pathsep)
    except KeyError:
        user_paths = []
    return top_n


def test_run():
    """Test count_words() with some inputs."""
    print count_words("cat bat mat cat bat cat", 3)
    print count_words("betty bought a bit of butter but the butter was bitter", 3)
    # print count_words("offer is secret click secret link secret sports link play sports today went play sports secret sports event sports is today sports costs money", 12)


if __name__ == '__main__':
    test_run()