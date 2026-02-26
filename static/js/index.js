window.HELP_IMPROVE_VIDEOJS = false;

$(document).ready(function() {
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");

    });

    var options = {
			slidesToScroll: 1,
			slidesToShow: 1,
			loop: true,
			infinite: true,
			autoplay: false,
			autoplaySpeed: 3000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    // Loop on each carousel initialized
    for(var i = 0; i < carousels.length; i++) {
    	// Add listener to  event
    	carousels[i].on('before:show', state => {
    		console.log(state);
    	});
    }

    // Access to bulmaCarousel instance of an element
    var element = document.querySelector('#my-element');
    if (element && element.bulmaCarousel) {
    	// bulmaCarousel instance is available as element.bulmaCarousel
    	element.bulmaCarousel.on('before-show', function(state) {
    		console.log(state);
    	});
    }

    var waveIntro = document.querySelector('.seacache-wave-intro');
    if (waveIntro) {
      var pointCount = 26;

      var clamp = function(value, min, max) {
        return Math.min(max, Math.max(min, value));
      };

      var createBaseProfile = function(mid, waveA, waveB) {
        var base = [];
        for (var i = 0; i < pointCount; i++) {
          var v = mid + Math.sin(i * 0.54) * waveA + Math.cos(i * 0.27) * waveB;
          base.push(v);
        }
        return base;
      };

      var createInitialHeights = function(base, jitter, min, max) {
        var heights = [];
        for (var i = 0; i < base.length; i++) {
          heights.push(clamp(base[i] + (Math.random() * jitter * 2 - jitter), min, max));
        }
        return heights;
      };

      var smoothHeights = function(values, passes) {
        var smoothed = values.slice();
        for (var p = 0; p < passes; p++) {
          var next = [];
          for (var i = 0; i < smoothed.length; i++) {
            var prevIndex = (i - 1 + smoothed.length) % smoothed.length;
            var nextIndex = (i + 1) % smoothed.length;
            next.push(
              smoothed[prevIndex] * 0.2 +
              smoothed[i] * 0.6 +
              smoothed[nextIndex] * 0.2
            );
          }
          smoothed = next;
        }
        return smoothed;
      };

      var buildClipPath = function(values) {
        var points = [];
        var step = 100 / (values.length - 1);
        for (var i = 0; i < values.length; i++) {
          points.push((step * i).toFixed(2) + '% ' + values[i].toFixed(1) + '%');
        }
        points.push('100% 100%');
        points.push('0 100%');
        return 'polygon(' + points.join(', ') + ')';
      };

      var applyHeights = function(frontHeights, backHeights) {
        waveIntro.style.setProperty('--wf-shape', buildClipPath(frontHeights));
        waveIntro.style.setProperty('--wb-shape', buildClipPath(backHeights));
      };

      var shiftRight = function(values) {
        if (values.length < 2) {
          return;
        }
        values.unshift(values.pop());
      };

      var frontBase = createBaseProfile(57, 3.5, 1.8);
      var backBase = createBaseProfile(60, 3.1, 1.5);
      var frontHeights = smoothHeights(createInitialHeights(frontBase, 2.8, 46, 72), 2);
      var backHeights = smoothHeights(createInitialHeights(backBase, 2.4, 49, 76), 2);

      applyHeights(frontHeights, backHeights);
      setInterval(function() {
        shiftRight(frontHeights);
        shiftRight(backHeights);
        applyHeights(frontHeights, backHeights);
      }, 150);
    }

})
