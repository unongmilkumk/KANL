package example.geneticAlgorithm.half

import kotlinAILibrary.GeneticAlgorithm
import kotlinAILibrary.NeuralNetwork
import kotlin.math.abs

fun main() {
    val mutationRate = 0.3
    val geneticAlgorithm = GeneticAlgorithm(List(300) { NeuralNetwork(1, 2, 1) },
        {a, _ ->
            a.memoList[0] = -800 * abs(a.forward(doubleArrayOf(1.0), false)[0] - 0.5) + 100
            -800 * abs(a.forward(doubleArrayOf(1.0), false)[0] - 0.5) + 100
        },
        {a, b -> a.sortedByDescending { b[a.indexOf(it)] }.take(a.size / 2)}, null, mutationRate)
    var a = doubleArrayOf()
    var bestAI = geneticAlgorithm.p[0]
    repeat(2000) { generation ->
        geneticAlgorithm.evolve()
        a = geneticAlgorithm.p.map { it.memoList[0] as Double }.toDoubleArray()
        bestAI = geneticAlgorithm.p[a.sortedDescending().withIndex().maxBy { it.value }.index]

        println("Generation $generation complete")
        println(bestAI.forward(doubleArrayOf(1.0), false)[0])
    }
    val n = geneticAlgorithm.p.indexOf(bestAI)
    println("Score : ${a[n]}")
    println("Result : ${bestAI.forward(doubleArrayOf(1.0), false)[0]}")
}