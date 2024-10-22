package example.kotris.a_library.swp

import java.awt.Color
import java.awt.Graphics
import java.awt.event.KeyListener
import javax.swing.JPanel

class SwpCanvas(color : Color) : JPanel() {
    private lateinit var paintcode : (g: Graphics) -> Unit
    init {
        background = color
        isFocusable = true
    }
    fun paint(code : (g: Graphics) -> Unit) {
        paintcode = code
    }
    override fun paint(g: Graphics) {
        super.paint(g)
        paintcode(g)
    }
    fun addKKeyListener(listener: KeyListener) {
        addKeyListener(listener)
    }
    fun rp() {
        repaint()
    }
}