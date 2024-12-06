Þ    6      Ì              |     }  "     ,   ½  .   ê  +        E  "   ^  "     )   ¤  *   Î  *   ù  +   $  3   P       ±        N  m   è     V  @   s     ´  )   Ã     í  
                +   @     l          §      Æ     ç     	     	     (	  Ì   A	  >   
  ,   M
  ,   z
  /   §
  -   ×
  &     *   ,  $   W     |       )     Õ   Ç  J        è     ô     û  8     2   @    s  "   ô  .     <   F  :     @   ¾     ÿ  -     Q   E  7     8   Ï  8     9   A  <   {  $   ¸  Î   Ý  ¿   ¬     l  8     b   =        8   ¹  $   ò       	   '  0   1  8   b  *     )   Æ  +   ð  -     (   J  '   s       '   ·  û   ß  D   Û  2      6   S  3     6   ¾  *   õ  2      +   S  !     !   ¡  +   Ã  #  ï  h        |            8   ¦  B   ß   **Callable** -- model class. **Callable** -- model layer class. **dict[str, float]** -- all metrics results. **list[Callable]** -- list of metrics classes. **list[Callable]** -- list of model layers. **tf.Tensor** -- output. Builds the following simple model. CNN (Convolutional Neural Network) Checks the :class:`BaseModel` parameters. Checks the :class:`SetupLayer` parameters. Checks the :class:`SetupModel` parameters. Checks the :class:`SimpleModel` parameters. Class variables whose values are available methods. Defines the base model. If you want to use some other settings, implement it as a method of this class. If you implemented, set the name as the ``func`` key in ``__init__`` and the method as the value. In eager mode, you can output calculation results using ``print`` or logging in :meth:`train_step`, :meth:`test_step`, or ``.call`` of class-form models. In graph mode, you can output too. But you will need to implement a custom layer that ``tf.print`` the input. MLP (Multi Layer Perceptron) Output gradients and update model parameters. (back propagation) Output losses. Output predictions. (forward propagation) Outputs the model predictions. Parameters Returns Returns list of metrics classes. Sets :class:`lib.model.simple.SimpleModel`. Sets ``keras.layers.Conv2D``. Sets ``keras.layers.Dense``. Sets ``keras.layers.Flatten``. Sets ``keras.layers.MaxPool2D``. Sets ``keras.layers.ReLU``. Sets up model layers. Sets up model. Sets up the model layer. Since the structure of a class-based model is not defined until input is given, ``.summary`` cannot be used. For the same reason, trained weights cannot be applied, so dummy data is input in ``__init__``. This function is decorated by ``@override`` and ``@property``. This function is decorated by ``@override``. This is the module that builds simple model. This is the module that defines the base model. This is the module that sets up model layers. This is the module that sets up model. This method is decorated by ``@override``. Trains the model one step at a time. Update metrics. Updates metrics. Validations the model one step at a time. When using ``.fit`` or ``.evaluate``, Metrics must be run ``.reset_state`` at the start of an epoch. By setting the return value of this method to a list of all metrics, it will automatically run ``.reset_state``. You can customize :meth:`train_step` and :meth:`test_step` using ``.fit``. class list. input. parameters. tuple of inputs and labels (and weights for each input). tuple of labels, preds, losses, and sample_weight. Project-Id-Version: template-tensorflow 
Report-Msgid-Bugs-To: 
POT-Creation-Date: 2024-12-06 21:46+0900
PO-Revision-Date: YEAR-MO-DA HO:MI+ZONE
Last-Translator: FULL NAME <EMAIL@ADDRESS>
Language: ja
Language-Team: ja <LL@li.org>
Plural-Forms: nplurals=1; plural=0;
MIME-Version: 1.0
Content-Type: text/plain; charset=utf-8
Content-Transfer-Encoding: 8bit
Generated-By: Babel 2.16.0
 **Callable** -- ã¢ãã«ã¯ã©ã¹ **Callable** -- ã¢ãã«ã¬ã¤ã¤ã¼ã¯ã©ã¹ **dict[str, float]** -- ãã¹ã¦ã®ã¡ããªã¯ã¹ã®çµæ **list[Callable]** -- ã¡ããªã¯ã¹ã¯ã©ã¹ã®ãªã¹ã **list[Callable]** -- ã¢ãã«ã¬ã¤ã¤ã¼ã¯ã©ã¹ã®ãªã¹ã **tf.Tensor** -- åºå æ¬¡ã®ã·ã³ãã«ã¢ãã«ãæ§ç¯ããã ç³ã¿è¾¼ã¿ãã¥ã¼ã©ã«ãããã¯ã¼ã¯ (CNN: Convolutional Neural Network) :class:`BaseModel` ã®ãã©ã¡ã¼ã¿ãç¢ºèªããã :class:`SetupLayer` ã®ãã©ã¡ã¼ã¿ãç¢ºèªããã :class:`SetupModel` ã®ãã©ã¡ã¼ã¿ãç¢ºèªããã :class:`SimpleModel` ã®ãã©ã¡ã¼ã¿ãç¢ºèªããã ä½¿ç¨å¯è½ãªã¡ã½ãããå¤ã«æã¤ã¯ã©ã¹å¤æ°ã ãã¼ã¹ã¢ãã«ãå®ç¾©ããã ä»ã®è¨­å®ãä½¿ç¨ãããå ´åããã®ã¯ã©ã¹ã®ã¡ã½ããã¨ãã¦å®è£ãããå®è£ããå ´åã``__init__`` ã® ``func`` ã®ã­ã¼ã«ååããå¤ã«ã¡ã½ãããå®è£ãããã¨ã eagerã¢ã¼ãã®éã:meth:`train_step` ã :meth:`test_step`ãã¯ã©ã¹å½¢å¼ã®ã¢ãã«ã® ``.call`` åã§ ``print`` ãloggingæ©è½ãä½¿ç¨ãã¦ãè¨ç®çµæãåºåã§ããã graphã¢ã¼ãã§ãåºåã§ããããå¥åã ``tf.print`` ãããããªã«ã¹ã¿ã ã¬ã¤ã¤ã¼ãå®è£ãããªã©å·¥å¤«ãå¿è¦ã§ããã å¤å±¤ãã¼ã»ããã­ã³(MLP: Multi Layer Perceptron) å¾éãç®åºããã¢ãã«ãã©ã¡ã¼ã¿ãæ´æ°ããã(èª¤å·®éä¼æ­: back propagation) èª¤å·®ãç®åºããã äºæ¸¬ãç®åºããã(é ä¼çª: forward propagation) ã¢ãã«ã®äºæ¸¬ãåºåããã ãã©ã¡ã¼ã¿ æ»ãå¤ ã¡ããªã¯ã¹ã¯ã©ã¹ã®ãªã¹ããè¿ãã :class:`lib.model.simple.SimpleModel` ãè¨­å®ããã ``keras.layers.Conv2D`` ãè¨­å®ããã ``keras.layers.Dense`` ãè¨­å®ããã ``keras.layers.Flatten`` ãè¨­å®ããã ``keras.layers.MaxPool2D`` ãè¨­å®ããã ``keras.layers.ReLU`` ãè¨­å®ããã ã¢ãã«ã¬ã¤ã¤ã¼ãè¨­å®ããã ã¢ãã«ãè¨­å®ããã ã¢ãã«ã¬ã¤ã¤ã¼ãè¨­å®ããã ã¯ã©ã¹å½¢å¼ã¢ãã«ã®æ§é ã¯å¥åãä¸ããããã¾ã§å®ç¾©ãããªãããã``.summary`` ãä½¿ç¨ã§ããªããåæ§ã«å­¦ç¿ããéã¿ãé©ç¨ãããã¨ãã§ããªãããã``__init__`` ã§ããã¼ãã¼ã¿ãä¸ããã ``@override`` ã¨ ``@property`` ã§ãã³ã¬ã¼ãããã¦ããã ``@override`` ã§ãã³ã¬ã¼ãããã¦ããã ã·ã³ãã«ã¢ãã«ãæ§ç¯ããã¢ã¸ã¥ã¼ã«ã ãã¼ã¹ã¢ãã«ãå®ç¾©ããã¢ã¸ã¥ã¼ã«ã ã¢ãã«ã¬ã¤ã¤ã¼ãè¨­å®ããã¢ã¸ã¥ã¼ã«ã ã¢ãã«ãè¨­å®ããã¢ã¸ã¥ã¼ã«ã ``@override`` ã§ãã³ã¬ã¼ãããã¦ããã 1ã¹ãããåã¢ãã«ãå­¦ç¿ããã ã¡ããªã¯ã¹ãæ´æ°ããã ã¡ããªã¯ã¹ãæ´æ°ããã 1ã¹ãããåã¢ãã«ãæ¤è¨¼ããã ``.fit`` ã ``.evaluate`` ãä½¿ç¨ããéãã¨ããã¯ã®éå§æã«ã¡ããªã¯ã¹ã¯ ``.reset_state`` ãå®è¡ããªããã°ãªããªããæ¬ã¡ã½ããã®æ»ãå¤ã«ãã¹ã¦ã®ã¡ããªã¯ã¹ã®ãªã¹ããè¨­å®ãããã¨ã§èªåã§ ``.reset_state`` ãå®è¡ãããã ``.fit`` ãä½¿ç¨ãã¤ã¤ã:meth:`train_step` ã¨ :meth:`test_step` ãã«ã¹ã¿ãã¤ãºã§ããã ã¯ã©ã¹ã®ä¸è¦§ å¥å ãã©ã¡ã¼ã¿ å¥åã¨ã©ãã«(ã¨å¥åãã¨ã®éã¿)ã®ã¿ãã« ã©ãã«ã¨äºæ¸¬ãèª¤å·®ãå¥åãã¨ã®éã¿ã®ã¿ãã«ã 