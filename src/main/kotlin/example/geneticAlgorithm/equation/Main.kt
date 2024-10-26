package example.geneticAlgorithm.equation

import kotlinAILibrary.GeneticAlgorithm
import kotlinAILibrary.NeuralNetwork
import kotlin.math.abs
import kotlin.math.min
import kotlin.math.sqrt
import kotlin.random.Random

@Suppress("UNCHECKED_CAST")
fun main() {
    val mutationRate = 0.07
    val geneticAlgorithm = GeneticAlgorithm(List(30000) { NeuralNetwork(2, listOf(2), 2) },
        {a, _ ->
            a.memoList.add(Triple(0.0, 0.0, Double.MIN_VALUE))
            a.memoList.add(Triple(1.0, 1.0, Double.MAX_VALUE))
            a.memoList.add(0.0)
            for (i1 in 10 until 20) {
                for (j1 in 0 until 10) {
                    val i = i1.toDouble()
                    val j = j1.toDouble()
                    val p = (i * i) - (4 * j)
                    if (p <= 0) continue
                    val b =  (sqrt(p) - i) / 2
                    val c =  (-sqrt(p) - i) / 2
                    var p1 = 0.0
                    p1 += -800 * abs(a.forward(doubleArrayOf(i, j), false)[0] - b) + 100
                    p1 += -800 * abs(a.forward(doubleArrayOf(i, j), false)[1] - c) + 100
                    if ((a.memoList[0] as Triple<Double, Double, Double>).third <= p1) a.memoList[0] = Triple(i, j, p1)
                    if ((a.memoList[1] as Triple<Double, Double, Double>).third >= p1) {
                        a.memoList[1] = Triple(i, j, p1)
                        a.memoList[2] = min(-800 * abs(a.forward(doubleArrayOf(i, j), false)[1] - c) + 100, -800 * abs(a.forward(doubleArrayOf(i, j), false)[0] - b) + 100)
                    }
                }
            }
            a.memoList[2].toString().toDouble()
        },
        {a, b -> a.sortedByDescending { b[a.indexOf(it)] }.take(a.size / 2)}, null, mutationRate)
    var a = doubleArrayOf()
    val bestAIs = arrayListOf<NeuralNetwork>()
    repeat(2000) { generation ->
        geneticAlgorithm.evolve()
        a = geneticAlgorithm.p.map { (it.memoList[0] as Triple<Double, Double, Double>).third }.toDoubleArray()
        bestAIs.add(geneticAlgorithm.p.maxByOrNull { (it.memoList[1] as Triple<Double, Double, Double>).third / 2 }!!)
        val bestAI = bestAIs.last()

        val i = (bestAI.memoList[0] as Triple<Double, Double, Double>).first
        val j = (bestAI.memoList[0] as Triple<Double, Double, Double>).second
        val p = (i * i) - (4 * j)
        val b =  (sqrt(p) - i) / 2
        val c =  (-sqrt(p) - i) / 2

        println("Generation $generation complete")
        println("Score : ${a[geneticAlgorithm.p.indexOf(bestAI)]} ${- ((a[geneticAlgorithm.p.indexOf(bestAI)] - 100.0) / 800.0)}")
        println("Best-Think : ${bestAI.forward(doubleArrayOf(i, j), false).joinToString(" ")}")
        println("Answer : $b $c")
    }
    val bestAI = bestAIs.maxByOrNull { (it.memoList[1] as Triple<Double, Double, Double>).third / 2 }!!
    val n = geneticAlgorithm.p.indexOf(bestAI)

    println("Score : ${a[n]}")
    for (i1 in 10 until 20) {
        for (j1 in 0 until 10) {
            val i = i1.toDouble()
            val j = j1.toDouble()
            val p = (i * i) - (4 * j)
            val b =  (sqrt(p) - i) / 2
            val c =  (-sqrt(p) - i) / 2
            println("Think : ${bestAI.forward(doubleArrayOf(i, j), false).joinToString(" ")}")
            println("Answer : $b $c")
        }
    }
}