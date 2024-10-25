package example.geneticAlgorithm.kotris.a_library.swp

import java.awt.event.ActionEvent
import java.awt.event.ActionListener
import javax.swing.JFrame
import javax.swing.Timer

class SwpWindow(title : String, private val width : Int, private val height : Int) : JFrame(title), ActionListener {
    val timer = Timer(1, this)
    var cv : SwpCanvas? = null

    fun enableWindow() {
        setSize(width, height)
        layout = null
        defaultCloseOperation = 3
        isVisible = true
    }

    fun setCanvas(canvas : SwpCanvas) {
        cv = canvas
        contentPane = cv
    }

    fun timerRun() {
        timer.start()
    }

    override fun actionPerformed(e: ActionEvent) {
        cv?.rp()
    }
}