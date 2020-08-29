package de.longuyen.data

enum class Dataset(val input: String, val feature: Int, val target: Int){
    TWITTER("data/tweets.csv", 5, 0),
    SMS("data/spam.csv", 1, 0)
}
