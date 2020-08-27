package de.longuyen.bayes

import kotlin.test.Test
import kotlin.test.assertEquals

class TestNaiveBayesClassifier{
    @Test
    fun testNaiveBayesClassifier(){
        val c = NaiveBayesClassifier<String, Int>()
        val d = mutableListOf(
                arrayOf("I", "like", "tesla"),
                arrayOf("I", "like", "kotlin"),
                arrayOf("I", "do not", "like", "tesla"),
                arrayOf("I", "do not", "like", "java"),
        ).toTypedArray()
        val t = arrayOf(
                1,
                1,
                0,
                0
        )
        c.learn(d, t)
        assertEquals(c.tCount[0], 2)
        assertEquals(c.tCount[1], 2)

        assertEquals(c.fCount["java"], 1)
        assertEquals(c.fCount["like"], 4)
        assertEquals(c.fCount["tesla"], 2)
        assertEquals(c.fCount["kotlin"], 1)
        assertEquals(c.fCount["I"], 4)
        assertEquals(c.fCount["do not"], 2)

        assertEquals(c.fTCount[1]!!["tesla"], 1)
        assertEquals(c.fTCount[1]!!["kotlin"], 1)
        assertEquals(c.fTCount[1]!!["like"], 2)
        assertEquals(c.fTCount[1]!!["I"], 2)

        assertEquals(c.fTCount[0]!!["tesla"], 1)
        assertEquals(c.fTCount[0]!!["java"], 1)
        assertEquals(c.fTCount[0]!!["java"], 1)
        assertEquals(c.fTCount[0]!!["do not"], 2)
        assertEquals(c.fTCount[0]!!["I"], 2)

        assertEquals(c.tFCount[0], 8)
        assertEquals(c.tFCount[1], 6)

        assertEquals(c.counts["dSum"]!!, 4)
        assertEquals(c.counts["fSum"]!!, 14)

        assertEquals(c.predict(arrayOf("I", "do not", "like", "painting")), 0)
        assertEquals(c.predict(arrayOf("I", "do", "like", "painting")), 1)
    }
}