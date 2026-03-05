//SOURCECODE_ORIGINAL_FILE_PATH=/Users/sethsull/Documents/coding/hands_on_scala/myScript.scala
//SOURCECODE_ORIGINAL_CODE_START_MARKER
def main(myArg: String, myOtherArg: Int) =
  println("hello" + " " + myArg)
  // println(myOtherArg + myArg)

@main
def hello() =
  println("hello" + " ")

type main = mainargs.main; private def myScript_scala_millScriptMainSelf = this; object MillScriptMain_myScript_scala { def main(args: Array[String]): Unit = this.getClass.getMethods.find(m => m.getName == "main" && m.getParameters.map(_.getType) == Seq(classOf[Array[String]]) && m.getReturnType == classOf[Unit]) match{ case Some(m) => m.invoke(myScript_scala_millScriptMainSelf, args); case None => mainargs.Parser(myScript_scala_millScriptMainSelf).runOrExit(args) }}