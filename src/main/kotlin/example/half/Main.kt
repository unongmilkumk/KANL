package example.half

import kotlinAILibrary.GeneticAlgorithm
import kotlinAILibrary.NeuralNetwork
import kotlin.math.abs
import kotlin.random.Random

fun main() {
    val mutationRate = 0.3
    val geneticAlgorithm = GeneticAlgorithm(List(300) { NeuralNetwork(1, 2, 1) },
        {a, _ ->
            a.memoList[0] = -800 * abs(a.forward(doubleArrayOf(1.0), false)[0] - 0.5) + 100
            -800 * abs(a.forward(doubleArrayOf(1.0), false)[0] - 0.5) + 100
        },
        {a, b -> a.sortedByDescending { b[a.indexOf(it)] }.take(a.size / 2)},
        {(parent1, parent2) ->
            val child = NeuralNetwork(1, 2, 1)
            for (j in 0 until 1) {
                child.wih[0][j] = if (Random.nextDouble() > 0.5) parent1.wih[0][j] else parent2.wih[0][j]
            }
            child.bho[0] = if (Random.nextDouble() > 0.5) parent1.bho[0] else parent2.bho[0]
            for (i in 0 until 1) {
                child.who[i][0] = if (Random.nextDouble() > 0.5) parent1.who[i][0] else parent2.who[i][0]
            }
            child.bih[0] = if (Random.nextDouble() > 0.5) parent1.bih[0] else parent2.bih[0]
            child.bih[1] = if (Random.nextDouble() > 0.5) parent1.bih[1] else parent2.bih[1]
            child},
        { nn ->
            for (j in 0 until 1) {
                if (Random.nextDouble() < mutationRate) {
                    nn.wih[0][j] += Random.nextDouble(-0.5, 0.5)
                }
            }
            for (j in 0 until 1) {
                if (Random.nextDouble() < mutationRate) {
                    nn.bih[j] += Random.nextDouble(-0.5, 0.5)
                }
            }
            for (i in 0 until 1) {
                if (Random.nextDouble() < mutationRate) {
                    nn.who[i][0] += Random.nextDouble(-0.5, 0.5)
                }
            }
            if (Random.nextDouble() < mutationRate) {
                nn.bho[0] += Random.nextDouble(-0.5, 0.5)
            }
            nn
        })
    var a = doubleArrayOf()
    var bestAI = geneticAlgorithm.p[0]
    var q: Pair<DoubleArray, NeuralNetwork>
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