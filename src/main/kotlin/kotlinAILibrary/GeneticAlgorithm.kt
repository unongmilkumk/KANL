package kotlinAILibrary

import kotlin.random.Random

class GeneticAlgorithm(population : List<NeuralNetwork>,
                       val score : (NeuralNetwork, List<NeuralNetwork>) -> (Double),
                       val select : (List<NeuralNetwork>, List<Double>) -> (List<NeuralNetwork>),
                       cross : ((Pair<NeuralNetwork, NeuralNetwork>) -> NeuralNetwork)?,
                       val mutate : (NeuralNetwork) -> NeuralNetwork
) {
    constructor(population : List<NeuralNetwork>,
                           score : (NeuralNetwork, List<NeuralNetwork>) -> (Double),
                           select : (List<NeuralNetwork>, List<Double>) -> (List<NeuralNetwork>),
                           cross : ((Pair<NeuralNetwork, NeuralNetwork>) -> NeuralNetwork)?,
                mutationRate : Double
    ) : this(population, score, select, cross, { nn ->
        val inputSize = population[0].input
        val hiddenSize = population[0].hidden
        val outputSize = population[0].output
        for (i in 0 until (inputSize - 1)) {
            for (j in 0 until (hiddenSize - 1)) {
                if (Random.nextDouble() < mutationRate) {
                    nn.wih[i][j] += Random.nextDouble(-0.5, 0.5)
                }
            }
        }
        for (j in 0 until (hiddenSize - 1)) {
            if (Random.nextDouble() < mutationRate) {
                nn.bih[j] += Random.nextDouble(-0.5, 0.5)
            }
        }
        for (i in 0 until (hiddenSize - 1)) {
            for (j in 0 until (outputSize - 1)) {
                if (Random.nextDouble() < mutationRate) {
                    nn.who[i][j] += Random.nextDouble(-0.5, 0.5)
                }
            }
        }
        for (j in 0 until (outputSize - 1)) {
            if (Random.nextDouble() < mutationRate) {
                nn.bho[j] += Random.nextDouble(-0.5, 0.5)
            }
        }
        nn
    })

    var p = population
    var pSize = population.size
    val fitnessScores = DoubleArray(pSize) { 0.0 }

    val crossover = cross
        ?: { (parent1, parent2) ->
            val inputSize = population[0].input
            val hiddenSize = population[0].hidden
            val outputSize = population[0].output
            val child = NeuralNetwork(inputSize, hiddenSize, outputSize)
            for (i in 0 until (inputSize - 1)) {
                for (j in 0 until (hiddenSize - 1)) {
                    child.wih[i][j] = if (Random.nextDouble() > 0.5) parent1.wih[i][j] else parent2.wih[i][j]
                }
            }

            for (i in 0 until (outputSize - 1)) {
                child.bho[i] = if (Random.nextDouble() > 0.5) parent1.bho[i] else parent2.bho[i]
            }

            for (i in 0 until (hiddenSize - 1)) {
                for (j in 0 until (outputSize - 1)) {
                    child.who[i][j] = if (Random.nextDouble() > 0.5) parent1.who[i][j] else parent2.who[i][j]
                }
            }

            for (i in 0 until (hiddenSize - 1)) {
                child.bih[i] = if (Random.nextDouble() > 0.5) parent1.bih[i] else parent2.bih[i]
            }
            child}

    fun startEvolve() {
        p.withIndex().forEach { (index, nn) ->
            fitnessScores[index] = score(nn, p)
        }
    }
    fun evolve() {
        val selected = select(p, fitnessScores.toList())

        val newPopulation = mutableListOf<NeuralNetwork>()
        for (i in 0 until pSize) {
            val parent1 = selected.random()
            val parent2 = selected.random()
            newPopulation.add(crossover(Pair(parent1, parent2)))
        }
        p = newPopulation.map { mutate(it) }

        p.withIndex().forEach { (index, nn) ->
            fitnessScores[index] = score(nn, p)
        }
    }
}