package kotlinAILibrary

import kotlin.math.exp
import kotlin.math.max
import kotlin.random.Random

class NeuralNetwork(
    val input: Int,
    val hiddenLayers: List<Int>,
    val output: Int,
    inputRange: IntRange = (0..1),
    outputRange: IntRange = (0..1),
    biasRange: IntRange = (0..1)
) {
    val weights = ArrayList<Array<DoubleArray>>()
    val biases = ArrayList<DoubleArray>()

    val memoList = arrayListOf<Any>()

    init {
        weights.add(Array(input) { DoubleArray(hiddenLayers[0]) { Random.nextDouble(inputRange.first.toDouble(), inputRange.last.toDouble()) } })
        biases.add(DoubleArray(hiddenLayers[0]) { Random.nextDouble(biasRange.first.toDouble(), biasRange.last.toDouble()) })

        for (i in 0 until hiddenLayers.size - 1) {
            weights.add(Array(hiddenLayers[i]) { DoubleArray(hiddenLayers[i + 1]) { Random.nextDouble(outputRange.first.toDouble(), outputRange.last.toDouble()) } })
            biases.add(DoubleArray(hiddenLayers[i + 1]) { Random.nextDouble(biasRange.first.toDouble(), biasRange.last.toDouble()) })
        }

        weights.add(Array(hiddenLayers.last()) { DoubleArray(output) { Random.nextDouble(outputRange.first.toDouble(), outputRange.last.toDouble()) } })
        biases.add(DoubleArray(output) { Random.nextDouble(biasRange.first.toDouble(), biasRange.last.toDouble()) })
    }

    constructor(a: List<Array<DoubleArray>>, b: List<DoubleArray>) : this(a[0].size, a.map { it[0].size }, b.last().size) {
        weights.clear()
        weights.addAll(a)
        biases.clear()
        biases.addAll(b)
    }

    fun memo(desc: Any, slot: Int): Any {
        while ((memoList.size - 1) < slot) {
            memoList.add("")
        }
        memoList[slot] = desc
        return desc
    }

    fun forward(input: DoubleArray, doSoftMax: Boolean = true): DoubleArray {
        var layerOutput = input

        for (i in weights.indices) {
            layerOutput = layerOutput.mm(weights[i]).pl(biases[i])
            if (i < weights.size - 1) {
                layerOutput = reLU(layerOutput)
            }
        }

        return if (doSoftMax) softmax(layerOutput) else layerOutput
    }

    private fun softmax(x: DoubleArray): DoubleArray {
        val exp = x.map { exp(it - x.max()) }
        return exp.map { it / exp.sum() }.toDoubleArray()
    }

    private fun reLU(x: DoubleArray): DoubleArray {
        return x.map { max(it, 0.0) }.toDoubleArray()
    }

    private fun DoubleArray.mm(weights: Array<DoubleArray>): DoubleArray {
        return DoubleArray(weights[0].size) { col ->
            this.withIndex().sumOf { (row, value) -> value * weights[row][col] }
        }
    }

    private fun DoubleArray.pl(biases: DoubleArray): DoubleArray {
        return this.withIndex().map { it.value + biases[it.index] }.toDoubleArray()
    }

    override fun equals(other: Any?): Boolean {
        return if (other is NeuralNetwork) {
            other.weights == weights && other.biases == biases
        } else false
    }
}
