# Region_Comparison_from_FourSquare
 Compare Denver metro to NC's Triangle using FourSquare data


When moving to, or visiting, a new city it can be difficult to decide where to focus one's attention. Here I create an algorithm which takes as its input a geographical region with which you are familiar as well as a second region about which you are curious. It outputs a pair of maps, one for each region, which are each shaded according to similarity based on data taken from the FourSquare database of nearby attractions. 

As a concrete example, consider two college towns: Chapel Hill, NC and Boulder, CO. Each town has a university region with nearby restaurants/shopping, these areas will be shaded the same color on both maps (in the example provided, they are red). Each town will have more everyday shopping (malls, grocery stores, big box stores, etc.), they will be shaded the same color (in the example provided, they are green). Using this tool you can use your intuition about one city to help make decisions regarding an entirely different city which you may have never visited.

Currently my script works only for the two above-mentioned cities. Chapel Hill is divided in to 102 sub-regions and Boulder in to 95 sub-regions. Using the FourSquare API I retrieve a list of all "venues" (stores, cafes, restaurants, schools, parks, etc.) near each sub-region. This leads to a total of 1,162 unique venues between the two cities, spanning 254 different category labels according to FourSquare. 

An unsupervised clustering algorithm is used to segment Chapel Hill (the city with which I am more familiar) into different groups according to the venue data, each group is shaded a particular color on the Chapel Hill map. Next, a K-Nearest-Neighbors classifier is trained on the (now labeled) Chapel Hill data then used to make predictions about how to group each of the 95 sub-regions in Boulder. These labels are then used to produce a map for the city of Boulder, which is very informative considering I just relocated here. 

For my project I propose an expansion of this script in the following ways:

1) Increase the geographical region covered. I would like to encompass the entire Denver metro area as well as all of North Carolina's Triangle region.
2) Create an interactive Dashboard with more functionality. Perhaps a user could select a region, or a series of regions, in North Carolina and the most-similar regions in Colorado would appear.
3) Increase the data on venues. Currently I consider only venue Category to make classifications. There is more data available from the FourSquare API that contains information about operating hours, peak hours, prices, ratings, customer reviews, customer check-ins, etc. This "premium" information is rate-limited to 500 calls per-day, however that should still be enough to scrape the data I need for the two regions. 
