package kotlinAILibrary

import kotlin.math.exp
import kotlin.math.max
import kotlin.random.Random

class NeuralNetwork(val input : Int, val hidden : Int, val output : Int, inputRange: IntRange = (0..1), outputRange: IntRange = (0..1), biasRange: IntRange = (0..1)) {
    var wih = arrayOf(doubleArrayOf())
    var who = arrayOf(doubleArrayOf())
    var bih = doubleArrayOf()
    var bho = doubleArrayOf()

    val memoList = arrayListOf<Any>()

    init {
        wih = Array(input) { DoubleArray(hidden) { Random.nextDouble(inputRange.first.toDouble(), inputRange.last.toDouble()) } }
        who = Array(hidden) { DoubleArray(output) { Random.nextDouble(outputRange.first.toDouble(), outputRange.last.toDouble()) } }
        bih = DoubleArray(hidden) { Random.nextDouble(biasRange.first.toDouble(), biasRange.last.toDouble()) }
        bho = DoubleArray(output) { Random.nextDouble(biasRange.first.toDouble(), biasRange.last.toDouble()) }
    }

    constructor(a : Array<DoubleArray>, b : Array<DoubleArray>, c : DoubleArray, d : DoubleArray) : this(a.size, b.size, b[0].size) {
        wih = a
        who = b
        bih = c
        bho = d
    }

    fun forward(input : DoubleArray, doSoftMax : Boolean = true): DoubleArray {
        var result = DoubleArray(hidden) {0.0}

        for ((index) in wih[0].withIndex()) {
            for ((index1, d) in input.withIndex()) {
                result[index] = result[index] + wih[index1][index] * d
            }
        }

        result = result.pl(bih)
        result = reLU(result)

        var result2 = DoubleArray(output) {0.0}

        for ((index) in who[0].withIndex()) {
            for ((index1, d) in result.withIndex()) {
                result2[index] = result2[index] + who[index1][index] * d
            }
        }

        result2 = result2.pl(bho)
        if (doSoftMax) result2 = softmax(result2)

        return result2
    }

    private fun softmax(x : DoubleArray) : DoubleArray {
        val exp = x.map { exp(it - x.max()) }
        return exp.map { it / exp.sum() }.toDoubleArray()
    }

    private fun reLU(x : DoubleArray) : DoubleArray {
        return x.map { max(it, 0.0) }.toDoubleArray()
    }

    private fun DoubleArray.pl(doubleArray: DoubleArray) : DoubleArray {
        return this.withIndex().map { it.value + doubleArray[it.index] }.toDoubleArray()
    }

    override fun equals(other: Any?): Boolean {
        return if (other is NeuralNetwork) {
            other.wih.contentEquals(wih) && other.who.contentEquals(who) && other.bih.contentEquals(bih) && other.bho.contentEquals(
                bho
            )
        } else false
    }
}