package de.longuyen.nlp

import org.apache.commons.csv.CSVFormat
import org.apache.commons.csv.CSVRecord
import java.io.FileReader


class IO {
    fun read() : Pair<Array<String>, Array<Int>>{
        val features = mutableListOf<String>()
        val targets = mutableListOf<Int>()
        val fr = FileReader("data/tweets.csv")
        val records: Iterable<CSVRecord> = CSVFormat.DEFAULT.parse(fr)
        for (record in records) {
            val target = record[0].toInt()
            val feature = record[5]
            features.add(feature)
            targets.add(target)
        }
        return Pair(features.toTypedArray(), targets.toTypedArray())
    }
}