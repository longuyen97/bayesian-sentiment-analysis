package de.longuyen.nlp

class Accuracy<T> : Metrics<T> {
    override fun compute(a: Array<T>, b: Array<T>) : Double {
        var ret = 0.0
        for(i in a.indices){
            if(a[i] == b[i]){
                ret += 1.0
            }
        }
        return ret / a.size.toDouble()
    }
}