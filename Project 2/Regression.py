import numpy as np

a = [1,2,3,4,5]
print np.mean(a)
print np.median(a)
print np.std(a)


# Compute the average number of bronze medals earned by countries who 
#     earned at least one gold medal.  

#         olympic_medal_counts = {'country_name':Series(countries),
#                             'gold': Series(gold),
#                             'silver': Series(silver),
#                             'bronze': Series(bronze)}
#     df = DataFrame(olympic_medal_counts)

#     df['bronze'].applymap(lambda x: x>= 1)

#     bronze_at_least_one_gold = df['bronze'][df['gold'] >= 1]
#     avg_bronze_at_least_one_gold = numpy.mean(bronze_at_least_one_gold)

#     df.apply(numpy.mean)
#     df['bronze'].applymap(lambda x: x>= 1)

#     #cuteness = pd.Series([1, 2, 3, 4, 5], index=['Cockroach', 'Fish', 'Mini Pig',
#                                                  'Puppy', 'Kitten'])
#     print "dear God, just appease the mini-pig crazed woman"
#     print cuteness[cuteness == 3]

for i in range(1,101):
    if i % 3 == 0 and i % 5 == 0:
        print "fizz buzz"
    elif i % 3 == 0:
        print "fizz"
    elif i % 5 == 0:
        print "buzz"
    else:
        print "bummer"