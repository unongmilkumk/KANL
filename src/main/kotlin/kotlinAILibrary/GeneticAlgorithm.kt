package kotlinAILibrary

import kotlin.random.Random

class GeneticAlgorithm(
    population: List<NeuralNetwork>,
    val score: (NeuralNetwork, List<NeuralNetwork>) -> Double,
    val select: (List<NeuralNetwork>, List<Double>) -> List<NeuralNetwork>,
    cross: ((Pair<NeuralNetwork, NeuralNetwork>) -> NeuralNetwork)?,
    val mutate: (NeuralNetwork) -> NeuralNetwork
) {
    constructor(
        population: List<NeuralNetwork>,
        score: (NeuralNetwork, List<NeuralNetwork>) -> Double,
        select: (List<NeuralNetwork>, List<Double>) -> List<NeuralNetwork>,
        cross: ((Pair<NeuralNetwork, NeuralNetwork>) -> NeuralNetwork)?,
        mutationRate: Double
    ) : this(population, score, select, cross, { nn ->
        nn.weights.forEachIndexed { layerIndex, layerWeights ->
            layerWeights.forEachIndexed { row, rowWeights ->
                rowWeights.indices.forEach { col ->
                    if (Random.nextDouble() < mutationRate) {
                        nn.weights[layerIndex][row][col] += Random.nextDouble(-0.5, 0.5)
                    }
                }
            }
        }
        nn.biases.forEachIndexed { layerIndex, layerBiases ->
            layerBiases.indices.forEach { i ->
                if (Random.nextDouble() < mutationRate) {
                    nn.biases[layerIndex][i] += Random.nextDouble(-0.5, 0.5)
                }
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
            val hiddenLayers = population[0].hiddenLayers
            val outputSize = population[0].output
            val child = NeuralNetwork(inputSize, hiddenLayers, outputSize)

            child.weights.forEachIndexed { layerIndex, layerWeights ->
                layerWeights.forEachIndexed { row, rowWeights ->
                    rowWeights.indices.forEach { col ->
                        child.weights[layerIndex][row][col] =
                            if (Random.nextDouble() > 0.5) parent1.weights[layerIndex][row][col]
                            else parent2.weights[layerIndex][row][col]
                    }
                }
            }

            child.biases.forEachIndexed { layerIndex, layerBiases ->
                layerBiases.indices.forEach { i ->
                    child.biases[layerIndex][i] =
                        if (Random.nextDouble() > 0.5) parent1.biases[layerIndex][i]
                        else parent2.biases[layerIndex][i]
                }
            }
            child
        }

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
