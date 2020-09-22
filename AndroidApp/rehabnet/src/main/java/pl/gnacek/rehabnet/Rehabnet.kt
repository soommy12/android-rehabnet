package pl.gnacek.rehabnet

import android.content.Context
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.examples.posenet.lib.BodyPart
import org.tensorflow.lite.examples.posenet.lib.Device
import org.tensorflow.lite.examples.posenet.lib.Person
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.exp

class Rehabnet(
    val context: Context,
//    val filename: String = "rehabnet_model.tflite"
    val filename: String = "rehabnet_img_model.tflite"
) {

    /** An Interpreter for the TFLite model.   */
    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    private val NUM_LITE_THREADS = 4
    private val currentExerciseFrames = mutableListOf<MutableMap<MutableList<Float>, MutableList<Float>>>()

    val exercises = hashMapOf(
        1 to "deep squat",
        2 to "hurdle step",
        3 to "inline lunge",
        4 to "side lunge",
        5 to "sit to stand",
        6 to "standing active straight leg raise",
        7 to "standing shoulder abduction",
        8 to "standing shoulder extension",
        9 to "standing shoulder internal-external rotation",
        10 to "standing shoulder scaption"
    )

    fun estimateExerciseForFrame(person: Person): Pair<String, Float> {
        val input = mutableListOf<Float>()
        val outputArray = initOutput(getInterpreter())
        val xValues = mutableListOf<Float>()
        val yValues = mutableListOf<Float>()
        loop@for(keyPoint in person.keyPoints) {
            when(keyPoint.bodyPart) {
                // do not take those joints, not needed for exercise estimation
                BodyPart.NOSE, BodyPart.LEFT_EYE, BodyPart.RIGHT_EYE, BodyPart.LEFT_EAR, BodyPart.RIGHT_EAR -> continue@loop
                else -> {
                    xValues.add(keyPoint.position.x.toFloat())
                    yValues.add(keyPoint.position.y.toFloat())
                }
            }
        }
        val xMax = xValues.max()!!.toDouble()
        val xMin = xValues.min()!!.toDouble()
        val yMax = yValues.max()!!.toDouble()
        val yMin = yValues.min()!!.toDouble()
        val sizeRefactor = 28
        val scale = kotlin.math.max((xMax - xMin), (yMax - yMin))
        for(idx in xValues.indices) {
            xValues[idx] = (((xValues[idx] - ((xMax + xMin)/2))/scale + 0.5)*sizeRefactor).toFloat()
            yValues[idx] = (((yValues[idx] - ((yMax + yMin)/2))/scale + 0.5)*sizeRefactor).toFloat()
        }
        input.addAll(xValues)
        input.addAll(yValues)
        val inputArray: Array<FloatArray> = arrayOf(
           input.toFloatArray()
        )
        getInterpreter().run(inputArray, outputArray)
        val classIdx = outputArray[0].indexOf(outputArray[0].max()!!)
        var probability = softmax(outputArray[0])[classIdx] * 100
        if(probability.isNaN()) probability = -1f
        return Pair(exercises[classIdx]!!, probability)
    }

    private fun initOutput(interpreter: Interpreter): Array<FloatArray> {

        val logitsShape = interpreter.getOutputTensor(0).shape()
        return Array(logitsShape[0]) {
            FloatArray(logitsShape[1])
        }
    }

    private fun initRgbOutput(interpreter: Interpreter): Array<FloatArray> {
        val logitsShape = interpreter.getOutputTensor(0).shape()
        return Array(logitsShape[0]) {
            FloatArray(logitsShape[1])
        }
    }

    private fun softmax(logits: FloatArray): FloatArray {
        val expLogits = FloatArray(logits.size)
        val softmaxed = FloatArray(logits.size)
        logits.forEachIndexed { idx, value -> expLogits[idx] = exp(value) }
        val expSum = expLogits.sum()
        expLogits.forEachIndexed { idx, value -> softmaxed[idx] = value/expSum}
        return softmaxed
    }

    /** Preload and memory map the model file, returning a MappedByteBuffer containing the model. */
    private fun loadModelFile(path: String, context: Context): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(path)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        return inputStream.channel.map(
            FileChannel.MapMode.READ_ONLY, fileDescriptor.startOffset, fileDescriptor.declaredLength
        )
    }

    private fun getInterpreter(): Interpreter {
        if (interpreter != null) {
            return interpreter!!
        }
        val options = Interpreter.Options()
        options.setNumThreads(NUM_LITE_THREADS)
        when (Device.CPU) {
            Device.CPU -> { }
            Device.GPU -> {
                gpuDelegate = GpuDelegate()
                options.addDelegate(gpuDelegate)
            }
            Device.NNAPI -> options.setUseNNAPI(true)
        }
        interpreter = Interpreter(loadModelFile(filename, context), options)
        return interpreter!!
    }

    fun saveFrame(person: Person) {
        val xValues = mutableListOf<Float>()
        val yValues = mutableListOf<Float>()
        val frameMap = mutableMapOf<MutableList<Float>, MutableList<Float>>()
        loop@for(keyPoint in person.keyPoints) {
            when(keyPoint.bodyPart) {
                // do not take those joints, not needed for exercise estimation
                BodyPart.NOSE, BodyPart.LEFT_EYE, BodyPart.RIGHT_EYE, BodyPart.LEFT_EAR, BodyPart.RIGHT_EAR -> continue@loop
                else -> {
                    xValues.add(keyPoint.position.x.toFloat())
                    yValues.add(keyPoint.position.y.toFloat())
                }
            }
        }
        val xMax = xValues.max()!!.toDouble()
        val xMin = xValues.min()!!.toDouble()
        val yMax = yValues.max()!!.toDouble()
        val yMin = yValues.min()!!.toDouble()
        val sizeRefactor = 255
        val scale = kotlin.math.max((xMax - xMin), (yMax - yMin))
        for(idx in xValues.indices) {
            xValues[idx] = (((xValues[idx] - ((xMax + xMin)/2))/scale + 0.5)*sizeRefactor).toFloat()
            yValues[idx] = (((yValues[idx] - ((yMax + yMin)/2))/scale + 0.5)*sizeRefactor).toFloat()
        }
        frameMap[xValues] = yValues
        currentExerciseFrames.add(frameMap)
    }

    fun estimateExercise() {
        currentExerciseFrames.forEachIndexed { index, mutableMap ->
            println("Frame $index")
            println("X: ${mutableMap.keys}")
            println("Y: ${mutableMap.values}")
        }
    }
}