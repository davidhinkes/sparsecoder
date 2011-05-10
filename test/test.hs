import AI.SparseCoder
import Data.Packed.Matrix
import Data.Packed.Vector
import Data.String.Utils
import Graphics.Transform.Magick.Images
import System
import qualified System.Random as R

import Debug.Trace

f x = 1.0 / ( 1.0 + exp(-x))
f' x = exp(x) / ( (1.0 + exp x) ^ 2 )

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


getImage :: String -> IO (Matrix Double)
getImage fn = do
  contents <- readFile fn
  let list_image = concat $ map (\y ->split "," y) (split "\n" contents)
  let binary_image = list_image `seq` map (\y -> read y :: Double) list_image
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

main = do
  args <- getArgs
  images <- mapM (\x -> getImage x) args
  samples <- mapM (\x -> selectSubImage 8 images) [1..10000]
  let sc = create 64 25 f f'
  let (sc', g) = trainN sc [] samples 10000
  putLst $ reverse g

putLst [] = return ()
putLst (x:xs) = do
  putStrLn $ show  x
  putLst xs

main2 = do
  let sc = create 10 7 f f'
  let xs = map createVector[1..100]
  sc' <- go sc xs 500
  putStrLn $ show $ squaredError sc' xs

go :: SparseCoder -> [Vector Double] -> Int -> IO SparseCoder
go sc _ 0 = return sc

go sc xs i = do
  let sc' = train sc xs
  let se = squaredError sc' xs
  --putStrLn $ show $ se
  se `seq` go sc' xs (i-1)
