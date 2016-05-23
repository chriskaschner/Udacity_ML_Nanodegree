import numpy as np

a = [1,2,3,4,5]
print np.mean(a)
print np.median(a)
print np.std(a)


Compute the average number of bronze medals earned by countries who 
    earned at least one gold medal.  

        olympic_medal_counts = {'country_name':Series(countries),
                            'gold': Series(gold),
                            'silver': Series(silver),
                            'bronze': Series(bronze)}
    df = DataFrame(olympic_medal_counts)

    df['bronze'].map(lambda x: x>= 1)

    bronze_at_least_one_gold = df['bronze'][df['gold'] >= 1]
    avg_bronze_at_least_one_gold = numpy.mean(bronze_at_least_one_gold)