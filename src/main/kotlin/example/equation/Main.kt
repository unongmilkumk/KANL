package example.equation

import kotlinAILibrary.GeneticAlgorithm
import kotlinAILibrary.NeuralNetwork
import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.random.Random

@Suppress("UNCHECKED_CAST")
fun main() {
    val mutationRate = 0.07
    val geneticAlgorithm = GeneticAlgorithm(List(30000) { NeuralNetwork(2, 2, 2) },
        {a, _ ->
            a.memoList.add(Triple(0.0, 0.0, Double.MIN_VALUE))
            a.memoList.add(Triple(1.0, 1.0, Double.MAX_VALUE))
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
                    if ((a.memoList[1] as Triple<Double, Double, Double>).third >= p1) a.memoList[1] = Triple(i, j, p1)
                }
            }
            (a.memoList[1] as Triple<Double, Double, Double>).third / 2
        },
        {a, b -> a.sortedByDescending { b[a.indexOf(it)] }.take(a.size / 2)},
        {(parent1, parent2) ->
            val child = NeuralNetwork(2, 2, 2)
            for (j in 0 until 1) {
                child.wih[0][j] = if (Random.nextDouble() > 0.5) parent1.wih[0][j] else parent2.wih[0][j]
                child.wih[1][j] = if (Random.nextDouble() > 0.5) parent1.wih[1][j] else parent2.wih[1][j]
            }
            child.bho[0] = if (Random.nextDouble() > 0.5) parent1.bho[0] else parent2.bho[0]
            child.bho[1] = if (Random.nextDouble() > 0.5) parent1.bho[1] else parent2.bho[1]
            for (i in 0 until 1) {
                child.who[i][0] = if (Random.nextDouble() > 0.5) parent1.who[i][0] else parent2.who[i][0]
                child.who[i][1] = if (Random.nextDouble() > 0.5) parent1.who[i][1] else parent2.who[i][1]
            }
            child.bih[0] = if (Random.nextDouble() > 0.5) parent1.bih[0] else parent2.bih[0]
            child.bih[1] = if (Random.nextDouble() > 0.5) parent1.bih[1] else parent2.bih[1]
            child},
        { nn ->
            for (j in 0 until 1) {
                if (Random.nextDouble() < mutationRate) {
                    nn.wih[0][j] += Random.nextDouble(-0.5, 0.5)
                    nn.wih[1][j] += Random.nextDouble(-0.5, 0.5)
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
                if (Random.nextDouble() < mutationRate) {
                    nn.who[i][1] += Random.nextDouble(-0.5, 0.5)
                }
            }
            if (Random.nextDouble() < mutationRate) {
                nn.bho[0] += Random.nextDouble(-0.5, 0.5)
            }
            if (Random.nextDouble() < mutationRate) {
                nn.bho[1] += Random.nextDouble(-0.5, 0.5)
            }
            nn
        })
    var a = doubleArrayOf()
    val bestAIs = arrayListOf<NeuralNetwork>()
    var q: Pair<DoubleArray, NeuralNetwork>
    repeat(2000) { generation ->
        q = geneticAlgorithm.evolve()
        a = q.first
        bestAIs.add(q.second)
        val bestAI = bestAIs.last()

        val i = (bestAI.memoList[0] as Triple<Double, Double, Double>).first
        val j = (bestAI.memoList[0] as Triple<Double, Double, Double>).second
        val p = (i * i) - (4 * j)
        val b =  (sqrt(p) - i) / 2
        val c =  (-sqrt(p) - i) / 2

        println("Generation $generation complete")
        println("Score : ${a[geneticAlgorithm.p.indexOf(bestAI)]}")
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