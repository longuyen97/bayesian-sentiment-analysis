package de.longuyen.data

import org.apache.commons.csv.CSVFormat
import org.apache.commons.csv.CSVRecord
import java.io.*


class IO(private val dataset: Dataset) : Serializable {
    companion object {
        private const val serialVersionUID: Long = -4270053884763734247
    }

    fun read(): Pair<Array<String>, Array<String>> {
        val features = mutableListOf<String>()
        val targets = mutableListOf<String>()
        val fr = FileReader(dataset.input)
        val records: Iterable<CSVRecord> = CSVFormat.DEFAULT.parse(fr)
        for (record in records) {
            val target = record[dataset.target]
            val feature = record[dataset.feature]
            features.add(feature)
            targets.add(target)
        }
        return Pair(features.toTypedArray(), targets.toTypedArray())
    }
}