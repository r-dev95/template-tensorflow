Þ    R      ¬              <     =  ,   Y       *     '   É     ñ  *   	  '   4  ;   \  ;     ,   Ô  )     )   +     U  3   æ  #     <   >  <   {  <   ¸  ³   õ     ©	     »	     Í	  -   Ù	     
  
   
  (   %
     N
     ^
  -   f
  $   
  '   ¹
      á
            :   !  W   \  D   ´  D   ù  w   >  @   ¶  <   ÷  >   4  9   s  <   ­  5   ê        '   .  )   V  )     %   ª  %   Ð  *   ö  =   !  Z   _     º     Ð     ì  
      -     #   9     ]      }           ¿     Æ     á     ù            :     [     b     }  5     5   Ë  3        5     A     S     `     q    v  "   ÷  9        T  &   l  #        ·  )   Ò  &   ü  6   #  6   Z  :     7   Ì  7     À   <  <   ý  *   :  C   e  C   ©  C   í  Ð   1            #     D  M   `  '   ®     Ö  9   æ        	   <  ;   F  0     4   ³  -   è          2  6   K  j     G   í  \   5  ]     B   ð  =   3  A   q  M   ³  P     ?   R       0   ®  /   ß  /     *   ?  *   j  2     l   È  Y   5   #      !   ³      Õ      å   ,   ø      %!     B!     ^!  *   z!     ¥!     ¬!     Â!     Õ!     ô!  0   "  	   D"     N"     g"  7   }"  7   µ"  5   í"     ##     3#     R#  %   b#     #   **Callable** -- data class. **Callable** -- data pipeline. (``tf.data``) **tf.Tensor** -- input. **tf.Tensor** -- input. (after preprocess) **tf.Tensor** -- input. (after process) **tf.Tensor** -- label. **tf.Tensor** -- label. (after preprocess) **tf.Tensor** -- label. (after process) **tf.train.Example** -- value of ``tf.train.Example`` type. **tf.train.Feature** -- value of ``tf.train.Feature`` type. Checks the :class:`BaseLoadData` parameters. Checks the :class:`Processor` parameters. Checks the :class:`SetupData` parameters. Child classes that inherit this class must set the list of file paths to ``params[K.FILES]`` before running ``super().__init__(params=params)``. Class variables whose values are available methods. Converts ``tf.train.Example`` type. Converts ``tf.train.Feature`` type. (``tf.train.BytesList``) Converts ``tf.train.Feature`` type. (``tf.train.FloatList``) Converts ``tf.train.Feature`` type. (``tf.train.Int64List``) If you want to use some other settings, implement it as a method of this class. If you implemented, set the name as the ``func`` key in ``__init__()`` and the method as the value. Loads Cifar data. Loads Mnist data. Loads data. Make a data pipeline to load a TFRecord data. Makes data loader. Parameters Parses one example from a TFRecord data. Processes data. Returns Run :meth:`lib.data.processor.Processor.run`. Run preprocess. (:meth:`preprocess`) Runs ``keras.layers.CategoryEncoding``. Runs ``keras.layers.Rescaling``. Runs preprocess. Runs process. Set the batch configuration. (``tf.data.Dataset.batch()``) Set the function to parse one example from a TFRecord data. (``tf.data.Dataset.map()``) Set the interleave configuration. (``tf.data.Dataset.interleave()``) Set the list of data file pathes. (``tf.data.Dataset.list_files()``) Set the parsing configuration according to the format in which the data was written. (``tf.io.parse_single_example()``) Set the prefetch configuration. (``tf.data.Dataset.prefetch()``) Set the repeat configuration. (``tf.data.Dataset.repeat()``) Set the shuffle configuration. (``tf.data.Dataset.shuffle()``) Sets :class:`lib.data.cifar.Cifar` (cifar10 or cifar100). Sets :class:`lib.data.mnist.Mnist` (mnist or fashion mnist). Sets the shape of the preprocessed inputs and labels. Sets up data. This is the module load and write data. This is the module that loads Cifar data. This is the module that loads Mnist data. This is the module that process data. This is the module that sets up data. This method is decorated by ``@override``. Used to process data when making a ``tf.data`` data pipeline. When writing TFRecord data, we make the elements one-dimensional, so we restore the shape. Writes TFRecord data. a set of inputs and labels. all number of data. file path. image size. (vertical x horizontal x channel) image size. (vertical x horizontal) input shape. (after preprocess) input shape. (before preprocess) input size. (elements per input) input. input. (before preprocess) input. (before process) label shape. (after preprocess) label shape. (before preprocess) label size. (elements per label) label. label. (before preprocess) label. (before process) one-dimensional list with elements of type ``bytes``. one-dimensional list with elements of type ``float``. one-dimensional list with elements of type ``int``. parameters. protocol massage. random seed. steps per epoch. type Project-Id-Version: template-tensorflow 
Report-Msgid-Bugs-To: 
POT-Creation-Date: 2024-11-23 15:50+0900
PO-Revision-Date: YEAR-MO-DA HO:MI+ZONE
Last-Translator: FULL NAME <EMAIL@ADDRESS>
Language: ja
Language-Team: ja <LL@li.org>
Plural-Forms: nplurals=1; plural=0;
MIME-Version: 1.0
Content-Type: text/plain; charset=utf-8
Content-Transfer-Encoding: 8bit
Generated-By: Babel 2.16.0
 **Callable** -- ãã¼ã¿ã¯ã©ã¹ **Callable** -- ãã¼ã¿ãã¤ãã©ã¤ã³ (``tf.data``) **tf.Tensor** -- å¥å **tf.Tensor** -- å¥å (åå¦çå¾) **tf.Tensor** -- å¥å (å¦çå¾) **tf.Tensor** -- ã©ãã« **tf.Tensor** -- ã©ãã« (åå¦çå¾) **tf.Tensor** -- ã©ãã« (å¦çå¾) **tf.train.Example** -- ``tf.train.Example`` åã®å¤ **tf.train.Feature** -- ``tf.train.Feature`` åã®å¤ :class:`BaseLoadData` ã®ãã©ã¡ã¼ã¿ãç¢ºèªããã :class:`Processor` ã®ãã©ã¡ã¼ã¿ãç¢ºèªããã :class:`SetupData` ã®ãã©ã¡ã¼ã¿ãç¢ºèªããã ãã®ã¯ã©ã¹ãç¶æ¿ããå­ã¯ã©ã¹ã¯ã``super().__init__(params=params)`` ãå®è¡ããåã«ãã¡ã¤ã«ãã¹ã®ãªã¹ãã ``params[K.FILES]`` ã«è¨­å®ããå¿è¦ãããã ä½¿ç¨å¯è½ãªã¡ã½ãããå¤ã«æã¤ã¯ã©ã¹å¤æ°ã ``tf.train.Example`` åã«å¤æããã ``tf.train.Feature`` åã«å¤æããã (``tf.train.BytesList``) ``tf.train.Feature`` åã«å¤æããã (``tf.train.FloatList``) ``tf.train.Feature`` åã«å¤æããã (``tf.train.Int64List``) ä»ã®è¨­å®ãä½¿ç¨ãããå ´åããã®ã¯ã©ã¹ã®ã¡ã½ããã¨ãã¦å®è£ãããå®è£ããå ´åã``__init__()`` ã® ``func`` ã®ã­ã¼ã«ååããå¤ã«ã¡ã½ãããå®è£ãããã¨ã Cifarãã¼ã¿ãèª­ã¿è¾¼ãã Mnistãã¼ã¿ãèª­ã¿è¾¼ãã ãã¼ã¿ãèª­ã¿è¾¼ãã TFRecordãã¼ã¿ãèª­ã¿è¾¼ããã¼ã¿ãã¤ãã©ã¤ã³ãä½æããã ãã¼ã¿ã­ã¼ãã¼ãä½æããã ãã©ã¡ã¼ã¿ TFRecordãã¼ã¿ãã1ã¬ã³ã¼ãããã¼ã¹ããã ãã¼ã¿ãå¦çããã æ»ãå¤ :meth:`lib.data.processor.Processor.run` ãå®è¡ããã åå¦çãå®è¡ããã (:meth:`preprocess`) ``keras.layers.CategoryEncoding`` ãå®è¡ããã ``keras.layers.Rescaling`` ãå®è¡ããã åå¦çãå®è¡ããã å¦çãå®è¡ããã ããããè¨­å®ãã (``tf.data.Dataset.batch()``) TFRecordãã¼ã¿ãã1ã¬ã³ã¼ãããã¼ã¹ããé¢æ°ãè¨­å®ããã (``tf.data.Dataset.map()``) ã¤ã³ã¿ãªã¼ããè¨­å®ããã (``tf.data.Dataset.interleave()``) ãã¼ã¿ãã¡ã¤ã«ãã¹ã®ãªã¹ããè¨­å®ããã (``tf.data.Dataset.list_files()``) æ¸ãè¾¼ã¾ãããã¼ã¿ã®ãã©ã¼ãããã«å¿ãã¦ããã¼ã¹ã®è¨­å®ãè¡ãã ããªãã§ããè¨­å®ããã (``tf.data.Dataset.prefetch()``) ãªãã¼ããè¨­å®ããã (``tf.data.Dataset.repeat()``) ã·ã£ããã«ãè¨­å®ããã (``tf.data.Dataset.shuffle()``) :class:`lib.data.cifar.Cifar` ãè¨­å®ããã (cifar10 ã¾ãã¯ cifar100) :class:`lib.data.mnist.Mnist` ãè¨­å®ããã (mnist ã¾ãã¯ fashion mnist) åå¦çãããå¥åã¨ã©ãã«ã®å½¢ç¶ãè¨­å®ããã ãã¼ã¿ãè¨­å®ããã ãã¼ã¿ãèª­ã¿æ¸ãããã¢ã¸ã¥ã¼ã«ã Cifarãã¼ã¿ãèª­ã¿è¾¼ãã¢ã¸ã¥ã¼ã«ã Mnistãã¼ã¿ãèª­ã¿è¾¼ãã¢ã¸ã¥ã¼ã«ã ãã¼ã¿ãå¦çããã¢ã¸ã¥ã¼ã«ã ãã¼ã¿ãè¨­å®ããã¢ã¸ã¥ã¼ã«ã ``@override`` ã§ãã³ã¬ã¼ãããã¦ããã ``tf.data`` ãã¼ã¿ãã¤ãã©ã¤ã³ãä½æããéããã¼ã¿ãå¦çããããã«ä½¿ç¨ããã TFRecordãæ¸ãè¾¼ãéãè¦ç´ ãä¸æ¬¡ååãããããå½¢ç¶ãåã«æ»ãã TFRecordãã¼ã¿ãæ¸ãè¾¼ãã å¥åã¨ã©ãã«ã®ã»ããã ç·ãã¼ã¿æ° ãã¡ã¤ã«ãã¹ ç»åãµã¤ãº (ç¸¦ Ã æ¨ª Ã ãã£ãã«) ç»åãµã¤ãº (ç¸¦ Ã æ¨ª) å¥åå½¢ç¶ (åå¦çå¾) å¥åå½¢ç¶ (åå¦çå) å¥åãµã¤ãº (å¥åãã¨ã®è¦ç´ æ°) å¥å å¥å (åå¦çå) å¥å (å¦çå) ã©ãã«å½¢ç¶ (åå¦çå¾) ã©ãã«å½¢ç¶ (åå¦çå) ã©ãã«ãµã¤ãº (ã©ãã«ãã¨ã®è¦ç´ æ°) ã©ãã« ã©ãã« (åå¦çå) ã©ãã« (å¦çå) ``bytes`` åã®è¦ç´ ãæã¤ä¸æ¬¡ååãªã¹ãã ``float`` åã®è¦ç´ ãæã¤ä¸æ¬¡ååãªã¹ãã ``int`` åã®è¦ç´ ãæã¤ä¸æ¬¡ååãªã¹ãã ãã©ã¡ã¼ã¿ ãã­ãã³ã«ã¡ãã»ã¼ã¸ ä¹±æ°ã·ã¼ã 1ã¨ããã¯ãã¨ã®ã¹ãããæ° å 