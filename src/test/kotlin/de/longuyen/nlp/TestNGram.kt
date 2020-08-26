package de.longuyen.nlp

import kotlin.test.Test
import kotlin.test.assertEquals

class TestNGram {
    @Test
    fun testNgram1(){
        val ngram1 = NGram(1)
        val result1 = ngram1.analyze("a")
        assertEquals(result1.size, 1)
        assertEquals(result1[0], "a")

        val result2 = ngram1.analyze("aa")
        assertEquals(result2.size, 1)
        assertEquals(result2[0], "aa")

        val result3 = ngram1.analyze("aa bb")
        assertEquals(result3.size, 2)
        assertEquals(result3[0], "aa")
        assertEquals(result3[1], "bb")

        val result4 = ngram1.analyze("aa bb cc")
        assertEquals(result4.size, 3)
        assertEquals(result4[0], "aa")
        assertEquals(result4[1], "bb")
        assertEquals(result4[2], "cc")
    }

    @Test
    fun testNgram2(){
        val ngram1 = NGram(2)
        val result3 = ngram1.analyze("aa bb")
        assertEquals(result3.size, 1)
        assertEquals(result3[0], "aa_bb")

        val result4 = ngram1.analyze("aa bb cc")
        assertEquals(result4.size, 2)
        assertEquals(result4[0], "aa_bb")
        assertEquals(result4[1], "bb_cc")
    }

    @Test
    fun testNgram3(){
        val ngram1 = NGram(3)
        val result3 = ngram1.analyze("aa bb cc")
        assertEquals(result3.size, 1)
        assertEquals(result3[0], "aa_bb_cc")

        val result4 = ngram1.analyze("aa bb cc dd")
        assertEquals(result4.size, 2)
        assertEquals(result4[0], "aa_bb_cc")
        assertEquals(result4[1], "bb_cc_dd")
    }
}