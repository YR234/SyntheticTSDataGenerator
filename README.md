# Synthetic Time series data generator
<h1 align="center">
  <br>
  <img src="https://github.com/YR234/SyntheticTSDataGenerator/blob/master/pictures/synthetic.png" alt="Synthetic" width="500">
</h1>
Synthetic Times series data generator is the open source code from the the paper "Generic transfer learning in time series classification using synthetic data generation"

## Install
Clone reposity, then:
```
$ pip install -r requirements.txt
```
# Usage
All of the code is well documented. </br>
We strongly reccomend to see our tutorial notebook for a simple explaination on how to use our generator.</br>
If you can't wait and want to run the generator allready, just run main.py and change variables according to your choise.

## Installation
[FR]: https://github.com/akashnimare/foco/releases

### OS X

1. Download [Foco-osx.x.x.x.dmg][FR] or [Foco-osx.x.x.x.zip][FR]
2. Open or unzip the file and drag the app into the `Applications` folder
3. Done!

### Windows
coming soon :stuck_out_tongue_closed_eyes:

### Linux

*Ubuntu, Debian 8+ (deb package):*

1. Download [Foco-linux.x.x.x.deb][FR]
2. Double click and install, or run `dpkg -i Foco-linux.x.x.x.deb` in the terminal
3. Start the app with your app launcher or by running `foco` in a terminal


### For developers
Clone the source locally:

```sh
$ git clone https://github.com/akashnimare/foco/
$ cd foco
```
If you're on Debian or Ubuntu, you'll also need to install
`nodejs-legacy`:

Use your package manager to install `npm`.

```sh
$ sudo apt-get install npm nodejs-legacy
```

Install project dependencies:

```sh
$ npm install
```
Start the app:

```sh
$ npm start
```

### Build installers

Build app for OSX
```sh
$ npm run build:osx
```
Build app for Linux
```sh
$ npm run build:linux
```

## Features

- [x] Offline support
- [x] Cross-platform
- [x] Awesome sounds
- [x] No singup/login required
- [ ] Auto launch
- [ ] Auto updates


## Usage

<kbd>Command/ctrl + R</kbd> - Reload

<kbd>command + q</kbd> - Quit App (while window is open).

## Built with
- [Electron](https://electron.atom.io)
- [Menubar](https://github.com/maxogden/menubar)

## Related
- [zulip-electron](https://github.com/zulip/zulip-electron)

## License

MIT  © [Akash Nimare](http://akashnimare.in)
