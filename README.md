# Proj. KANL
### Kotlin AI Neural Library

## What is it?
- AI library for Kotlin
- Easy way to access Neural Network

## How to import?
It doesn't support maven/gradle now, so just copy-paste to use on your project.

## How to code?
### Neural Network
```kotlin
NeuralNetwork(val input: Int,
val hiddenLayers: List<Int>,
val output: Int,
val inputRange: IntRange = (0..1),
val outputRange: IntRange = (0..1),
val biasRange: IntRange = (0..1))

NeuralNetwork(val weight: List<Array<DoubleArray>>, 
val bias: List<DoubleArray>)

NeuralNetwork.forward(input : DoubleArray, doSoftMax : Boolean = true) : DoubleArray
NeuralNetwork.memo(desc : Any, slot : Int) : Any

```
### Genetic Algorithm
```kotlin
GeneticAlgorithm(population : List<NeuralNetwork>, 
val score : (NeuralNetwork, List<NeuralNetwork>) -> (Double),
val select : (List<NeuralNetwork>, List<Double>) -> (List<NeuralNetwork>),
val cross : ((Pair<NeuralNetwork, NeuralNetwork>) -> NeuralNetwork)?, /**null -> default cross function*/.
val mutate : (NeuralNetwork) -> NeuralNetwork /**Double -> default mutation function with mutation rate*/
))

GeneticAlgorithm.startEvolve() /**before repeat**/
GeneticAlgorithm.evolve() /**during repeat**/
```

#### Example
```kotlin
import kotlinAILibrary.GeneticAlgorithm
import kotlinAILibrary.NeuralNetwork
import kotlin.math.abs

fun main() {
    val mutationRate = 0.3
    val geneticAlgorithm = GeneticAlgorithm(List(300) { NeuralNetwork(1, listOf(2), 1) },
        {a, _ ->
            a.memo(-800 * abs(a.forward(doubleArrayOf(1.0), false)[0] - 0.5) + 100).toString().toDouble()
        },
        {a, b -> a.sortedByDescending { b[a.indexOf(it)] }.take(a.size / 2)}, null, mutationRate)
    var a = doubleArrayOf()
    var bestAI = geneticAlgorithm.p[0]
    geneticAlgorithm.startEvolve()
    repeat(2000) { generation ->
        a = geneticAlgorithm.p.map { it.memoList[0] as Double }.toDoubleArray()
        bestAI = geneticAlgorithm.p[a.sortedDescending().withIndex().maxBy { it.value }.index]
        geneticAlgorithm.evolve()
    }
    val n = geneticAlgorithm.p.indexOf(bestAI)
    println("Score : ${a[n]}")
    println("Result : ${bestAI.forward(doubleArrayOf(1.0), false)[0]}")
}
```