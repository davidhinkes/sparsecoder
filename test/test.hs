import AI.SparseCoder
import Data.Maybe
import Data.Packed.Matrix
import Data.Packed.Vector
import qualified Data.String.Utils as U
import Graphics.Transform.Magick.Images
import System
import qualified System.Random as R
import qualified Data.ByteString.Char8 as B
import Data.ByteString.Lex.Double

f x = 1.0 / ( 1.0 + exp(-x))
f' x = let expx = exp x
           expxPlusOne = 1.0+expx
       in expx / ( expxPlusOne * expxPlusOne )

createVector:: Int -> Vector Double
createVector x = fromList $ map (\y -> fromIntegral (x `mod` 10) / 10.0) [1..10]

getVectors :: [FilePath] -> IO ([Vector Double])
getVectors fs = do
  initializeMagick
  getVectors' fs

getVectors' :: [FilePath] -> IO([Vector Double])
getVectors' [] = return []
getVectors' (f:fs) = do
  img <- readImage f
  getVectors' fs

toDouble x = let (d, _) = fromJust (readDouble  x) in d

getImage :: String -> IO (Matrix Double)
getImage fn = do
  contents <- B.readFile fn
  let list_image = concat $ map (\y -> B.split ',' y) (B.split '\n' contents)
  let binary_image = map (\y -> toDouble y) list_image
  let matrix_image = (512><512) binary_image
  return matrix_image

r :: IO Double
r = do
  g <- R.getStdGen
  let (x, g') = R.random g
  R.setStdGen g'
  return (x)

choose :: (Int, Int) -> IO Int
choose (a,b) = do
  rand <- r
  let d = fromIntegral (b - a) :: Double 
  let delta = round (d*rand)
  return (a + (fromIntegral delta))

selectSubImage :: Int -> [Matrix Double] -> IO(Vector Double)
selectSubImage p ms = do
  i_offset <- choose ( 0, 512 - p)
  j_offset <- choose ( 0, 512 - p)
  img <- choose(0, length(ms)-1)
  let r1 = (i_offset,j_offset) 
  let r2 = (p,p) 
  return $ flatten $ subMatrix r1 r2 (ms !! img)

imageize :: Int -> [a] -> [[a]]
imageize _ [] = [[]]
imageize i xs = (take i xs) : (imageize i (drop i xs))

outputHiddenLayer i m =
  let m' = subMatrix (0,0) (rows m, cols m -1) m
      raw_data = map (\x -> show x) $  toList $ flatten $ m'
      img = imageize i raw_data
      img_as_string = map (\x -> U.join " " x) img in
    U.join "\n" img_as_string

main = do
  args <- getArgs
  images <- mapM (\x -> getImage x) args
  samples <- mapM (\x -> selectSubImage 8 (images `seq` images)) [1..1000]
  let sc = create 64 25 f f'
  let (sc', g) = trainN sc [] samples 500
  putStrLn $ outputHiddenLayer 8 $ hiddenLayer sc'

putLst [] = return ()
putLst (x:xs) = do
  putStrLn $ show  x
  putLst xs
