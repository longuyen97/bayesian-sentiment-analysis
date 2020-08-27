package de.longuyen.nlp

import org.apache.commons.csv.CSVFormat
import org.apache.commons.csv.CSVRecord
import java.io.*


class IO  : Serializable{
    companion object {
        private const val serialVersionUID: Long = -4270053884763734247
    }

    fun read() : Pair<Array<String>, Array<Int>>{
        val features = mutableListOf<String>()
        val targets = mutableListOf<Int>()
        if(File("target/cache.ser").exists()){
            println("Cache of data found. Read cache.")

            FileInputStream(File("target/cache.ser")).use{
                ObjectInputStream(it).use{ois ->
                    return ois.readObject() as Pair<Array<String>, Array<Int>>
                }
            }
        }else{
            println("Cache of data NOT found. Creating new cache")

            val fr = FileReader("data/tweets.csv")
            val records: Iterable<CSVRecord> = CSVFormat.DEFAULT.parse(fr)
            for (record in records) {
                val target = record[0].toInt()
                val feature = record[5]
                features.add(feature)
                targets.add(target)
            }
            val ret = Pair(features.toTypedArray(), targets.toTypedArray())
             FileOutputStream("target/cache.ser").use { fos ->
                 ObjectOutputStream(fos).use { oos ->
                     oos.writeObject(ret)
                 }
             }
            return ret
        }
    }
}