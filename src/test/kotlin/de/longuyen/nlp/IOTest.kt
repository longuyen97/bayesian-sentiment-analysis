package de.longuyen.nlp

import kotlin.test.Test
import kotlin.test.assertEquals

class IOTest {
    @Test
    fun testReadIO(){
        val io = IO()
        val result = io.read()
        assertEquals(result.first.size, result.second.size)

        val targets = HashSet<Int>()
        for(i in result.second){
            targets.add(i)
        }
        assertEquals(targets.size, 2)

        var positiveCount = 0
        var negativeCount = 0
        for(i in result.second){
            if(i == 0){
                negativeCount += 1
            }else if(i == 4){
                positiveCount += 1
            }
        }
        assertEquals(positiveCount, negativeCount)
    }
}