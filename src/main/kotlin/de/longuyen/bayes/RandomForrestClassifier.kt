package de.longuyen.bayes

class RandomForrestClassifier<F, T>(private val count: Int = 5) : BayesianClassifier<F, T> {
    private val forrest = mutableListOf<BayesianClassifier<F, T>>()

    init {
        for(i in 0 until count){

        }
    }

    override fun initialize(documents: Array<Array<F>>, targets: Array<T>) {
    }

    override fun predict(document: Array<F>): T {
        TODO("Not yet implemented")
    }

    override fun predict(documents: Array<Array<F>>): List<T> {
        TODO("Not yet implemented")
    }

}